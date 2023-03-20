"""
Tests for functions in lex.py.
"""

import pytest
from piss import lex
from piss.lex import TokenKindVariant
from textwrap import dedent
from typing import Type


@pytest.mark.parametrize(
    ("input", "expected_token", "expected_chars_read"),
    [
        ("F", TokenKindVariant.Identifier("F"), 1),
        ("Foo", TokenKindVariant.Identifier("Foo"), 3),
        ("Foo_bar", TokenKindVariant.Identifier("Foo_bar"), 7),
        ("struct", TokenKindVariant.Keyword(lex.KeywordKind.STRUCT), 6),
    ],
)
def test_tokenize_identifier_or_keyword(
    input: str,
    expected_token: TokenKindVariant.Identifier | TokenKindVariant.Keyword,
    expected_chars_read: int,
) -> None:
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
) -> None:
    with pytest.raises(expected_exception):
        lex.tokenize_identifier_or_keyword(input),


@pytest.mark.parametrize(
    ("input", "expected_token", "expected_chars_read"),
    [
        ("1", TokenKindVariant.Integer(1), 1),
        ("123456789", TokenKindVariant.Integer(123456789), 9),
        ("123456789asdfghjkl", TokenKindVariant.Integer(123456789), 9),
    ],
)
def test_tokenize_integer(
    input: str, expected_token: lex.Token, expected_chars_read: int
) -> None:
    token, chars_read = lex.tokenize_integer(input)

    assert token == expected_token
    assert chars_read == expected_chars_read


@pytest.mark.parametrize(("input"), ["asdfghjkl"])
def test_fail_tokenize_integer(input: str) -> None:
    with pytest.raises(lex.LexError):
        lex.tokenize_integer(input)


@pytest.mark.parametrize(
    ("input", "expected_chars_read"),
    [
        (" \t\n\r123", 4),
        ("Hellow, World!", 0),
    ],
)
def test_whitespace(input: str, expected_chars_read: int) -> None:
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
def test_comments(input: str, expected_chars_read: int) -> None:
    chars_read = lex.skip_comments(input)
    assert chars_read == expected_chars_read


@pytest.mark.parametrize(
    ("input", "expected_token", "expected_length"),
    [
        ("1234", TokenKindVariant.Integer(1234), 4),
        ("=", TokenKindVariant.Equals(), 1),
        ("{", TokenKindVariant.LeftBrace(), 1),
        ("}", TokenKindVariant.RightBrace(), 1),
        (";", TokenKindVariant.SemiColon(), 1),
        (",", TokenKindVariant.Comma(), 1),
    ],
)
def test_token(input: str, expected_token: lex.TokenKind, expected_length: int) -> None:
    result = lex.token(input)

    assert result is not None

    kind: lex.TokenKind
    length: int

    kind, length = result
    assert kind == expected_token
    assert length == expected_length


def test_tokenize() -> None:
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
        TokenKindVariant.Keyword(lex.KeywordKind.STRUCT),
        TokenKindVariant.Identifier("PersonType"),
        TokenKindVariant.LeftBrace(),
        TokenKindVariant.Identifier("NameType"),
        TokenKindVariant.Identifier("name"),
        TokenKindVariant.Comma(),
        TokenKindVariant.Identifier("AgeType"),
        TokenKindVariant.Identifier("age"),
        TokenKindVariant.Comma(),
        TokenKindVariant.Identifier("HeightType"),
        TokenKindVariant.Identifier("height"),
        TokenKindVariant.Comma(),
        TokenKindVariant.RightBrace(),
        TokenKindVariant.SemiColon(),
    ]

    tokens = lex.tokenize(struct_example)
    token_kinds = [token.kind for token in tokens]

    assert token_kinds == expected_tokens_kinds
