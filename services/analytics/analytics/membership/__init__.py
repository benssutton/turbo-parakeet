"""Set membership / containment (multi-set scope). See Membership for semantics."""

from analytics.base import lazy_attributes
from analytics.membership.base import BloomMembership, Membership
from analytics.membership.exact import MembershipExact
from analytics.membership.rust import BloomRust

REFERENCE = "MembershipExact"
IMPLEMENTATIONS = ("BloomRust", "BloomFastbloom", "MembershipExact")

__getattr__ = lazy_attributes(__name__, {"BloomFastbloom": ".fastbloom"})
__all__ = [
    "Membership", "BloomMembership", "BloomRust", "BloomFastbloom", "MembershipExact",
    "REFERENCE", "IMPLEMENTATIONS",
]
