"""
Tests for functions in lex.py.
"""

import pytest
import piss.lex as lex
from textwrap import dedent
from typing import Type


@pytest.mark.parametrize(
    ("input", "expected_token", "expected_chars_read"),
    [
        ("F", lex.Identifier("F"), 1),
        ("Foo", lex.Identifier("Foo"), 3),
        ("Foo_bar", lex.Identifier("Foo_bar"), 7),
        ("struct", lex.Keyword(lex.KeywordKind.STRUCT), 6),
    ],
)
def test_tokenize_identifier_or_keyword(
    input: str, expected_token: lex.Identifier | lex.Keyword, expected_chars_read: int
):
    ident_or_keyword, chars_read = lex.tokenize_identifier_or_keyword(input)

    assert ident_or_keyword == expected_token
    assert chars_read == expected_chars_read


@pytest.mark.parametrize(
    ("input", "expected_exception"),
    [
        ("7Foo_bar", lex.UnexpectedCharacter),
        (".Foo_bar", lex.NoMatch),
        ("", lex.UnexpectedEOF),
    ],
)
def test_fail_tokenize_identifier_or_keyword(
    input: str, expected_exception: Type[lex.LexError]
):
    with pytest.raises(expected_exception) as exception_type:
        lex.tokenize_identifier_or_keyword(input),


@pytest.mark.parametrize(
    ("input", "expected_token", "expected_chars_read"),
    [
        ("1", lex.Integer(1), 1),
        ("123456789", lex.Integer(123456789), 9),
        ("123456789asdfghjkl", lex.Integer(123456789), 9),
    ],
)
def test_tokenize_integer(
    input: str, expected_token: lex.Token, expected_chars_read: int
):
    token, chars_read = lex.tokenize_integer(input)

    assert token == expected_token
    assert chars_read == expected_chars_read


@pytest.mark.parametrize(("input"), ["asdfghjkl"])
def test_fail_tokenize_integer(input: str):
    with pytest.raises(lex.LexError):
        lex.tokenize_integer(input)


@pytest.mark.parametrize(
    ("input", "expected_chars_read"),
    [
        (" \t\n\r123", 4),
        ("Hellow, World!", 0),
    ],
)
def test_whitespace(input: str, expected_chars_read: int):
    chars_read = lex.skip_whitespace(input)
    assert chars_read == expected_chars_read


@pytest.mark.parametrize(
    ("input", "expected_chars_read"),
    [
        ("// foo bar { baz }\n 1234", 19),
        ("{ baz \n 1234} hello wor\nld", 0),
        (" foo bar { baz } // a comment", 0),
    ],
)
def test_comments(input: str, expected_chars_read: int):
    chars_read = lex.skip_comments(input)
    assert chars_read == expected_chars_read


@pytest.mark.parametrize(
    ("input", "expected_token", "expected_length"),
    [
        ("1234", lex.Integer(1234), 4),
        ("=", lex.Equals(), 1),
        ("{", lex.LeftBrace(), 1),
        ("}", lex.RightBrace(), 1),
        (";", lex.SemiColon(), 1),
        (",", lex.Comma(), 1),
    ],
)
def test_token(input, expected_token, expected_length):
    token, length = lex.token(input)
    assert token == expected_token
    assert length == expected_length


def test_tokenize():
    struct_example = dedent(
        """
    struct PersonType {
        NameType name,
        AgeType age,
        HeightType height,
    };
        """
    )

    expected_tokens_kinds = [
        lex.Keyword(lex.KeywordKind.STRUCT),
        lex.Identifier("PersonType"),
        lex.LeftBrace(),
        lex.Identifier("NameType"),
        lex.Identifier("name"),
        lex.Comma(),
        lex.Identifier("AgeType"),
        lex.Identifier("age"),
        lex.Comma(),
        lex.Identifier("HeightType"),
        lex.Identifier("height"),
        lex.Comma(),
        lex.RightBrace(),
        lex.SemiColon(),
    ]

    tokens = lex.tokenize(struct_example)
    token_kinds = [token.kind for token in tokens]

    assert token_kinds == expected_tokens_kinds
