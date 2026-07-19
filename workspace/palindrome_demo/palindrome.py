"""
Utility for palindrome checks.
"""

from __future__ import annotations


def is_palindrome(s: str, ignore_case: bool = True, ignore_non_alnum: bool = True) -> bool:
    """
    Check if the given string s is a palindrome.

    By default, ignores case and non-alphanumeric characters.
    - ignore_case: if True, compare in lower-case.
    - ignore_non_alnum: if True, skip characters that are not alphanumeric.
    """
    if s is None:
        return False

    if ignore_case:
        s = s.lower()
    if ignore_non_alnum:
        s = ''.join(ch for ch in s if ch.isalnum())

    return s == s[::-1]

__all__ = ["is_palindrome"]
