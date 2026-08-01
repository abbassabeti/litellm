#### What this does ####
#   identifies lowest tpm deployment
import random
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

import httpx

import litellm
from litellm import token_counter
from litellm._logging import verbose_logger, verbose_router_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.core_helpers import _get_parent_otel_span_from_kwargs
from litellm.types.router import RouterErrors
from litellm.types.utils import LiteLLMPydanticObjectBase, StandardLoggingPayload
from litellm.utils import get_utc_datetime, print_verbose

from .base_routing_strategy import BaseRoutingStrategy

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    Span = Union[_Span, Any]
else:
    Span = Any

RateLimitMetric = Literal["rpm", "tpm"]


@dataclass(frozen=True, slots=True)
class SlidingWindowCacheKeys:
    """
    Cache keys + weight needed to compute a sliding-window effective count
    from two fixed-window (per wall-clock-minute) counters.
    """

    current_key: str
    previous_key: str
    previous_weight: float


def _build_sliding_window_keys(
    model_id: str | None,
    deployment_name: str | None,
    metric: RateLimitMetric,
    dt: datetime,
) -> SlidingWindowCacheKeys:
    """
    Sliding window counter algorithm (as used by Cloudflare, Kong, etc.):
    approximates a rolling window using the current and previous fixed-minute
    counters, weighted by how much of the previous minute's window still
    overlaps the trailing 60s from `dt`.
    """
    current_minute = dt.strftime("%H-%M")
    previous_minute = (dt - timedelta(minutes=1)).strftime("%H-%M")
    seconds_into_current_minute = dt.second + dt.microsecond / 1_000_000
    previous_weight = 1 - (seconds_into_current_minute / 60)
    return SlidingWindowCacheKeys(
        current_key=f"{model_id}:{deployment_name}:{metric}:{current_minute}",
        previous_key=f"{model_id}:{deployment_name}:{metric}:{previous_minute}",
        previous_weight=previous_weight,
    )


def _effective_sliding_window_count(
    current_count: float | None,
    previous_count: float | None,
    previous_weight: float,
) -> float:
    return (current_count or 0) + (previous_count or 0) * previous_weight


def _get_nested(mapping: Mapping[str, Any], outer_key: str, inner_key: str) -> Any:
    inner = mapping.get(outer_key)
    return inner.get(inner_key) if isinstance(inner, Mapping) else None


def _get_deployment_rate_limit(deployment: Mapping[str, Any], limit_key: RateLimitMetric) -> float:
    """
    A deployment's configured rpm/tpm limit can be set at 3 different levels,
    checked in this order; falls back to unlimited if none are set.
    """
    for candidate in (
        deployment.get(limit_key),
        _get_nested(deployment, "litellm_params", limit_key),
        _get_nested(deployment, "model_info", limit_key),
    ):
        if candidate is not None:
            return candidate
    return float("inf")


class RoutingArgs(LiteLLMPydanticObjectBase):
    ttl: int = 1 * 60  # 1min (RPM/TPM expire key)


class LowestTPMLoggingHandler_v2(BaseRoutingStrategy, CustomLogger):
    """
    Updated version of TPM/RPM Logging.

    Meant to work across instances.

    Caches individual models, not model_groups

    Uses batch get (redis.mget)

    Increments tpm/rpm limit using redis.incr
    """

    test_flag: bool = False
    logged_success: int = 0
    logged_failure: int = 0
    default_cache_time_seconds: int = 1 * 60 * 60  # 1 hour

    def __init__(self, router_cache: DualCache, routing_args: dict = {}):
        self.router_cache = router_cache
        self.routing_args = RoutingArgs(**routing_args)
        BaseRoutingStrategy.__init__(
            self,
            dual_cache=router_cache,
            should_batch_redis_writes=True,
            default_sync_interval=0.1,
        )

    def pre_call_check(self, deployment: Dict) -> Optional[Dict]:
        """
        Pre-call check + update model rpm

        Returns - deployment

        Raises - RateLimitError if deployment over defined RPM limit
        """
        try:
            # ------------
            # Setup values
            # ------------

            dt = get_utc_datetime()
            model_id = deployment.get("model_info", {}).get("id")
            deployment_name = deployment.get("litellm_params", {}).get("model")
            sliding_window_keys = _build_sliding_window_keys(
                model_id=model_id, deployment_name=deployment_name, metric="rpm", dt=dt
            )
            rpm_key = sliding_window_keys.current_key

            # check local result first (both current + previous minute buckets)
            local_current = self.router_cache.get_cache(key=rpm_key, local_only=True)
            local_previous = self.router_cache.get_cache(key=sliding_window_keys.previous_key, local_only=True)
            local_result = _effective_sliding_window_count(
                local_current, local_previous, sliding_window_keys.previous_weight
            )

            deployment_rpm = _get_deployment_rate_limit(deployment, "rpm")

            if local_result >= deployment_rpm:
                raise litellm.RateLimitError(
                    message="Deployment over defined rpm limit={}. current usage={}".format(
                        deployment_rpm, local_result
                    ),
                    llm_provider="",
                    model=deployment.get("litellm_params", {}).get("model"),
                    response=httpx.Response(
                        status_code=429,
                        content="{} rpm limit={}. current usage={}. id={}, model_group={}. Get the model info by calling 'router.get_model_info(id)".format(
                            RouterErrors.user_defined_ratelimit_error.value,
                            deployment_rpm,
                            local_result,
                            model_id,
                            deployment.get("model_name", ""),
                        ),
                        request=httpx.Request(
                            method="tpm_rpm_limits",
                            url="https://github.com/BerriAI/litellm",
                        ),  # type: ignore
                    ),
                )
            else:
                # if local result below limit, check redis ## prevent unnecessary redis checks

                result = self.router_cache.increment_cache(key=rpm_key, value=1, ttl=self.routing_args.ttl)
                effective_result = _effective_sliding_window_count(
                    result, local_previous, sliding_window_keys.previous_weight
                )
                if effective_result > deployment_rpm:
                    raise litellm.RateLimitError(
                        message="Deployment over defined rpm limit={}. current usage={}".format(
                            deployment_rpm, effective_result
                        ),
                        llm_provider="",
                        model=deployment.get("litellm_params", {}).get("model"),
                        response=httpx.Response(
                            status_code=429,
                            content="{} rpm limit={}. current usage={}".format(
                                RouterErrors.user_defined_ratelimit_error.value,
                                deployment_rpm,
                                effective_result,
                            ),
                            request=httpx.Request(
                                method="tpm_rpm_limits",
                                url="https://github.com/BerriAI/litellm",
                            ),  # type: ignore
                        ),
                    )
            return deployment
        except Exception as e:
            if isinstance(e, litellm.RateLimitError):
                raise e
            return deployment  # don't fail calls if eg. redis fails to connect

    async def async_pre_call_check(self, deployment: Dict, parent_otel_span: Optional[Span]) -> Optional[Dict]:
        """
        Pre-call check + update model rpm
        - Used inside semaphore
        - raise rate limit error if deployment over limit

        Why? solves concurrency issue - https://github.com/BerriAI/litellm/issues/2994

        Returns - deployment

        Raises - RateLimitError if deployment over defined RPM limit
        """
        try:
            # ------------
            # Setup values
            # ------------
            dt = get_utc_datetime()
            model_id = deployment.get("model_info", {}).get("id")
            deployment_name = deployment.get("litellm_params", {}).get("model")

            sliding_window_keys = _build_sliding_window_keys(
                model_id=model_id, deployment_name=deployment_name, metric="rpm", dt=dt
            )
            rpm_key = sliding_window_keys.current_key

            # check local result first (both current + previous minute buckets)
            local_current, local_previous = await self.router_cache.async_batch_get_cache(
                keys=[rpm_key, sliding_window_keys.previous_key], local_only=True
            )
            local_result = _effective_sliding_window_count(
                local_current, local_previous, sliding_window_keys.previous_weight
            )

            deployment_rpm = _get_deployment_rate_limit(deployment, "rpm")
            if local_result >= deployment_rpm:
                raise litellm.RateLimitError(
                    message="Deployment over defined rpm limit={}. current usage={}".format(
                        deployment_rpm, local_result
                    ),
                    llm_provider="",
                    model=deployment.get("litellm_params", {}).get("model"),
                    response=httpx.Response(
                        status_code=429,
                        content="{} rpm limit={}. current usage={}".format(
                            RouterErrors.user_defined_ratelimit_error.value,
                            deployment_rpm,
                            local_result,
                        ),
                        headers={"retry-after": str(60)},  # type: ignore
                        request=httpx.Request(
                            method="tpm_rpm_limits",
                            url="https://github.com/BerriAI/litellm",
                        ),  # type: ignore
                    ),
                    num_retries=deployment.get("num_retries"),
                )
            else:
                # if local result below limit, check redis ## prevent unnecessary redis checks
                result = await self._increment_value_in_current_window(key=rpm_key, value=1, ttl=self.routing_args.ttl)
                effective_result = _effective_sliding_window_count(
                    result, local_previous, sliding_window_keys.previous_weight
                )
                if effective_result > deployment_rpm:
                    raise litellm.RateLimitError(
                        message="Deployment over defined rpm limit={}. current usage={}".format(
                            deployment_rpm, effective_result
                        ),
                        llm_provider="",
                        model=deployment.get("litellm_params", {}).get("model"),
                        response=httpx.Response(
                            status_code=429,
                            content="{} rpm limit={}. current usage={}".format(
                                RouterErrors.user_defined_ratelimit_error.value,
                                deployment_rpm,
                                effective_result,
                            ),
                            headers={"retry-after": str(60)},  # type: ignore
                            request=httpx.Request(
                                method="tpm_rpm_limits",
                                url="https://github.com/BerriAI/litellm",
                            ),  # type: ignore
                        ),
                        num_retries=deployment.get("num_retries"),
                    )
            return deployment
        except Exception as e:
            if isinstance(e, litellm.RateLimitError):
                raise e
            return deployment  # don't fail calls if eg. redis fails to connect

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            """
            Update TPM/RPM usage on success
            """
            standard_logging_object: Optional[StandardLoggingPayload] = kwargs.get("standard_logging_object")
            if standard_logging_object is None:
                raise ValueError("standard_logging_object not passed in.")
            model_group = standard_logging_object.get("model_group")
            model = standard_logging_object["hidden_params"].get("litellm_model_name")
            id = standard_logging_object.get("model_id")
            if model_group is None or id is None or model is None:
                return
            elif isinstance(id, int):
                id = str(id)

            total_tokens = standard_logging_object.get("total_tokens")

            # ------------
            # Setup values
            # ------------
            dt = get_utc_datetime()
            current_minute = dt.strftime("%H-%M")  # use the same timezone regardless of system clock

            tpm_key = f"{id}:{model}:tpm:{current_minute}"
            # ------------
            # Update usage
            # ------------
            # update cache

            ## TPM
            self.router_cache.increment_cache(key=tpm_key, value=total_tokens, ttl=self.routing_args.ttl)
            ### TESTING ###
            if self.test_flag:
                self.logged_success += 1
        except Exception as e:
            verbose_logger.exception(
                "litellm.proxy.hooks.lowest_tpm_rpm_v2.py::log_success_event(): Exception occured - {}".format(str(e))
            )
            pass

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            """
            Update TPM usage on success
            """
            standard_logging_object: Optional[StandardLoggingPayload] = kwargs.get("standard_logging_object")
            if standard_logging_object is None:
                raise ValueError("standard_logging_object not passed in.")
            model_group = standard_logging_object.get("model_group")
            model = standard_logging_object["hidden_params"]["litellm_model_name"]
            id = standard_logging_object.get("model_id")
            if model_group is None or id is None:
                return
            elif isinstance(id, int):
                id = str(id)
            total_tokens = standard_logging_object.get("total_tokens")
            # ------------
            # Setup values
            # ------------
            dt = get_utc_datetime()
            current_minute = dt.strftime("%H-%M")  # use the same timezone regardless of system clock

            tpm_key = f"{id}:{model}:tpm:{current_minute}"
            # ------------
            # Update usage
            # ------------
            # update cache
            parent_otel_span = _get_parent_otel_span_from_kwargs(kwargs)
            ## TPM
            await self.router_cache.async_increment_cache(
                key=tpm_key,
                value=total_tokens,
                ttl=self.routing_args.ttl,
                parent_otel_span=parent_otel_span,
            )

            ### TESTING ###
            if self.test_flag:
                self.logged_success += 1
        except Exception as e:
            verbose_logger.exception(
                "litellm.proxy.hooks.lowest_tpm_rpm_v2.py::async_log_success_event(): Exception occured - {}".format(
                    str(e)
                )
            )
            pass

    def _return_potential_deployments(
        self,
        healthy_deployments: List[Dict],
        all_deployments: Dict,
        input_tokens: int,
        rpm_dict: Dict,
    ):
        lowest_tpm = float("inf")
        potential_deployments = []  # if multiple deployments have the same low value
        deployment_lookup = {
            deployment.get("model_info", {}).get("id"): deployment for deployment in healthy_deployments
        }
        for item, item_tpm in all_deployments.items():
            ## get the item from model list
            item = item.split(":")[0]
            _deployment = deployment_lookup.get(item)
            if _deployment is None:
                continue  # skip to next one
            elif item_tpm is None:
                continue  # skip if unhealthy deployment

            _deployment_tpm = None
            if _deployment_tpm is None:
                _deployment_tpm = _deployment.get("tpm")
            if _deployment_tpm is None:
                _deployment_tpm = _deployment.get("litellm_params", {}).get("tpm")
            if _deployment_tpm is None:
                _deployment_tpm = _deployment.get("model_info", {}).get("tpm")
            if _deployment_tpm is None:
                _deployment_tpm = float("inf")

            _deployment_rpm = None
            if _deployment_rpm is None:
                _deployment_rpm = _deployment.get("rpm")
            if _deployment_rpm is None:
                _deployment_rpm = _deployment.get("litellm_params", {}).get("rpm")
            if _deployment_rpm is None:
                _deployment_rpm = _deployment.get("model_info", {}).get("rpm")
            if _deployment_rpm is None:
                _deployment_rpm = float("inf")
            if item_tpm + input_tokens > _deployment_tpm:
                continue
            elif (
                (rpm_dict is not None and item in rpm_dict)
                and rpm_dict[item] is not None
                and (rpm_dict[item] + 1 >= _deployment_rpm)
            ):
                continue
            elif item_tpm == lowest_tpm:
                potential_deployments.append(_deployment)
            elif item_tpm < lowest_tpm:
                lowest_tpm = item_tpm
                potential_deployments = [_deployment]
        return potential_deployments

    def _common_checks_available_deployment(
        self,
        model_group: str,
        healthy_deployments: list,
        tpm_keys: list,
        tpm_values: Optional[list],
        rpm_keys: list,
        rpm_values: Optional[list],
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
    ) -> Optional[dict]:
        """
        Common checks for get available deployment, across sync + async implementations
        """

        if tpm_values is None or rpm_values is None:
            return None

        tpm_dict = {}  # {model_id: 1, ..}
        for idx, key in enumerate(tpm_keys):
            tpm_dict[tpm_keys[idx].split(":")[0]] = tpm_values[idx]

        rpm_dict = {}  # {model_id: 1, ..}
        for idx, key in enumerate(rpm_keys):
            rpm_dict[rpm_keys[idx].split(":")[0]] = rpm_values[idx]

        try:
            input_tokens = token_counter(messages=messages, text=input)
        except Exception:
            input_tokens = 0
        verbose_router_logger.debug(f"input_tokens={input_tokens}")
        # -----------------------
        # Find lowest used model
        # ----------------------

        if tpm_dict is None:  # base case - none of the deployments have been used
            # initialize a tpm dict with {model_id: 0}
            tpm_dict = {}
            for deployment in healthy_deployments:
                tpm_dict[deployment["model_info"]["id"]] = 0
        else:
            for d in healthy_deployments:
                ## if healthy deployment not yet used
                tpm_key = d["model_info"]["id"]
                if tpm_key not in tpm_dict or tpm_dict[tpm_key] is None:
                    tpm_dict[tpm_key] = 0

        all_deployments = tpm_dict
        potential_deployments = self._return_potential_deployments(
            healthy_deployments=healthy_deployments,
            all_deployments=all_deployments,
            input_tokens=input_tokens,
            rpm_dict=rpm_dict,
        )
        print_verbose("returning picked lowest tpm/rpm deployment.")

        if len(potential_deployments) > 0:
            return random.choice(potential_deployments)
        else:
            return None

    async def async_get_available_deployments(
        self,
        model_group: str,
        healthy_deployments: list,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
    ):
        """
        Async implementation of get deployments.

        Reduces time to retrieve the tpm/rpm values from cache
        """
        # get list of potential deployments
        verbose_router_logger.debug(
            f"get_available_deployments - Usage Based. model_group: {model_group}, healthy_deployments: {healthy_deployments}"
        )

        dt = get_utc_datetime()

        deployment_ids_and_names: tuple[tuple[Any, Any], ...] = tuple(
            (_get_nested(m, "model_info", "id"), _get_nested(m, "litellm_params", "model"))
            for m in healthy_deployments
            if isinstance(m, dict)
        )
        deployment_sliding_keys: tuple[tuple[SlidingWindowCacheKeys, SlidingWindowCacheKeys], ...] = tuple(
            (
                _build_sliding_window_keys(model_id=model_id, deployment_name=deployment_name, metric="tpm", dt=dt),
                _build_sliding_window_keys(model_id=model_id, deployment_name=deployment_name, metric="rpm", dt=dt),
            )
            for model_id, deployment_name in deployment_ids_and_names
        )

        tpm_keys = [tpm.current_key for tpm, _ in deployment_sliding_keys]
        rpm_keys = [rpm.current_key for _, rpm in deployment_sliding_keys]
        previous_tpm_keys = [tpm.previous_key for tpm, _ in deployment_sliding_keys]
        previous_rpm_keys = [rpm.previous_key for _, rpm in deployment_sliding_keys]

        n = len(tpm_keys)
        combined_keys = tpm_keys + rpm_keys + previous_tpm_keys + previous_rpm_keys

        combined_values = await self.router_cache.async_batch_get_cache(keys=combined_keys)  # [1, 2, None, ..]

        if combined_values is not None:
            tpm_values = [
                _effective_sliding_window_count(current, previous, tpm.previous_weight)
                for current, previous, (tpm, _) in zip(
                    combined_values[:n], combined_values[2 * n : 3 * n], deployment_sliding_keys
                )
            ]
            rpm_values = [
                _effective_sliding_window_count(current, previous, rpm.previous_weight)
                for current, previous, (_, rpm) in zip(
                    combined_values[n : 2 * n], combined_values[3 * n :], deployment_sliding_keys
                )
            ]
        else:
            tpm_values = None
            rpm_values = None

        deployment = self._common_checks_available_deployment(
            model_group=model_group,
            healthy_deployments=healthy_deployments,
            tpm_keys=tpm_keys,
            tpm_values=tpm_values,
            rpm_keys=rpm_keys,
            rpm_values=rpm_values,
            messages=messages,
            input=input,
        )

        try:
            assert deployment is not None
            return deployment
        except Exception:
            ### GET THE DICT OF TPM / RPM + LIMITS PER DEPLOYMENT ###
            deployment_dict = {}
            for index, _deployment in enumerate(healthy_deployments):
                if isinstance(_deployment, dict):
                    id = _deployment.get("model_info", {}).get("id")
                    _deployment_tpm = _get_deployment_rate_limit(_deployment, "tpm")

                    ### GET CURRENT TPM ###
                    current_tpm = tpm_values[index] if tpm_values else 0

                    _deployment_rpm = _get_deployment_rate_limit(_deployment, "rpm")

                    ### GET CURRENT RPM ###
                    current_rpm = rpm_values[index] if rpm_values else 0

                    deployment_dict[id] = {
                        "current_tpm": current_tpm,
                        "tpm_limit": _deployment_tpm,
                        "current_rpm": current_rpm,
                        "rpm_limit": _deployment_rpm,
                    }
            raise litellm.RateLimitError(
                message=f"{RouterErrors.no_deployments_available.value}. Passed model={model_group}. Deployments={deployment_dict}",
                llm_provider="",
                model=model_group,
                response=httpx.Response(
                    status_code=429,
                    content="",
                    headers={"retry-after": str(60)},  # type: ignore
                    request=httpx.Request(
                        method="tpm_rpm_limits",
                        url="https://github.com/BerriAI/litellm",
                    ),  # type: ignore
                ),
            )

    def get_available_deployments(
        self,
        model_group: str,
        healthy_deployments: list,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        parent_otel_span: Optional[Span] = None,
    ):
        """
        Returns a deployment with the lowest TPM/RPM usage.
        """
        # get list of potential deployments
        verbose_router_logger.debug(
            f"get_available_deployments - Usage Based. model_group: {model_group}, healthy_deployments: {healthy_deployments}"
        )

        dt = get_utc_datetime()

        deployment_ids_and_names: tuple[tuple[Any, Any], ...] = tuple(
            (_get_nested(m, "model_info", "id"), _get_nested(m, "litellm_params", "model"))
            for m in healthy_deployments
            if isinstance(m, dict)
        )
        deployment_sliding_keys: tuple[tuple[SlidingWindowCacheKeys, SlidingWindowCacheKeys], ...] = tuple(
            (
                _build_sliding_window_keys(model_id=model_id, deployment_name=deployment_name, metric="tpm", dt=dt),
                _build_sliding_window_keys(model_id=model_id, deployment_name=deployment_name, metric="rpm", dt=dt),
            )
            for model_id, deployment_name in deployment_ids_and_names
        )

        tpm_keys = [tpm.current_key for tpm, _ in deployment_sliding_keys]
        rpm_keys = [rpm.current_key for _, rpm in deployment_sliding_keys]
        previous_tpm_keys = [tpm.previous_key for tpm, _ in deployment_sliding_keys]
        previous_rpm_keys = [rpm.previous_key for _, rpm in deployment_sliding_keys]

        current_tpm_values = self.router_cache.batch_get_cache(
            keys=tpm_keys, parent_otel_span=parent_otel_span
        )  # [1, 2, None, ..]
        current_rpm_values = self.router_cache.batch_get_cache(keys=rpm_keys, parent_otel_span=parent_otel_span)
        previous_tpm_values = self.router_cache.batch_get_cache(
            keys=previous_tpm_keys, parent_otel_span=parent_otel_span
        )
        previous_rpm_values = self.router_cache.batch_get_cache(
            keys=previous_rpm_keys, parent_otel_span=parent_otel_span
        )

        if (
            current_tpm_values is not None
            and current_rpm_values is not None
            and previous_tpm_values is not None
            and previous_rpm_values is not None
        ):
            tpm_values = [
                _effective_sliding_window_count(current, previous, tpm.previous_weight)
                for current, previous, (tpm, _) in zip(current_tpm_values, previous_tpm_values, deployment_sliding_keys)
            ]
            rpm_values = [
                _effective_sliding_window_count(current, previous, rpm.previous_weight)
                for current, previous, (_, rpm) in zip(current_rpm_values, previous_rpm_values, deployment_sliding_keys)
            ]
        else:
            tpm_values = None
            rpm_values = None

        deployment = self._common_checks_available_deployment(
            model_group=model_group,
            healthy_deployments=healthy_deployments,
            tpm_keys=tpm_keys,
            tpm_values=tpm_values,
            rpm_keys=rpm_keys,
            rpm_values=rpm_values,
            messages=messages,
            input=input,
        )

        try:
            assert deployment is not None
            return deployment
        except Exception:
            ### GET THE DICT OF TPM / RPM + LIMITS PER DEPLOYMENT ###
            deployment_dict = {}
            for index, _deployment in enumerate(healthy_deployments):
                if isinstance(_deployment, dict):
                    id = _deployment.get("model_info", {}).get("id")
                    _deployment_tpm = _get_deployment_rate_limit(_deployment, "tpm")

                    ### GET CURRENT TPM ###
                    current_tpm = tpm_values[index] if tpm_values else 0

                    _deployment_rpm = _get_deployment_rate_limit(_deployment, "rpm")

                    ### GET CURRENT RPM ###
                    current_rpm = rpm_values[index] if rpm_values else 0

                    deployment_dict[id] = {
                        "current_tpm": current_tpm,
                        "tpm_limit": _deployment_tpm,
                        "current_rpm": current_rpm,
                        "rpm_limit": _deployment_rpm,
                    }
            raise ValueError(
                f"{RouterErrors.no_deployments_available.value}. Passed model={model_group}. Deployments={deployment_dict}"
            )
