"""Set membership / containment (multi-set scope). See Membership for semantics."""

from turbo_parakeet.techniques.base import lazy_attributes
from turbo_parakeet.techniques.membership.base import BloomMembership, Membership
from turbo_parakeet.techniques.membership.exact import MembershipExact
from turbo_parakeet.techniques.membership.rust import BloomRust

REFERENCE = "MembershipExact"
IMPLEMENTATIONS = ("BloomRust", "BloomFastbloom", "MembershipExact")

__getattr__ = lazy_attributes(__name__, {"BloomFastbloom": ".fastbloom"})
__all__ = [
    "Membership",
    "BloomMembership",
    "BloomRust",
    "BloomFastbloom",
    "MembershipExact",
    "REFERENCE",
    "IMPLEMENTATIONS",
]
