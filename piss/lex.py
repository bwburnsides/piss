"""
Tools for tokenizing the PISS lexicon.
"""

from dataclasses import dataclass
import enum
from typing import Callable, Literal


class KeywordKind(enum.Enum):
    """
    KeywordKind enumerates the keywords which are legal in PISS IDL.
    """

    CONST = "const"
    STRUCT = "struct"
    ENUM = "enum"
    TYPEDEF = "typedef"
    MODULE = "module"
    UINT = "uint"
    INT = "int"


@enum.unique
class TokenKindTag(enum.Enum):
    """
    TokenKindTag enumerates the types of Tokens which are legal in PISS IDL.
    """

    KEYWORD = enum.auto()
    IDENTIFIER = enum.auto()
    INTEGER = enum.auto()
    LEFT_BRACE = enum.auto()
    RIGHT_BRACE = enum.auto()
    LEFT_BRACKET = enum.auto()
    RIGHT_BRACKET = enum.auto()
    SEMICOLON = enum.auto()
    COMMA = enum.auto()
    EQUALS = enum.auto()


class TokenKindVariant:
    @dataclass
    class Keyword:
        keyword: KeywordKind
        tag: Literal[TokenKindTag.KEYWORD] = TokenKindTag.KEYWORD

    @dataclass
    class Identifier:
        name: str
        tag: Literal[TokenKindTag.IDENTIFIER] = TokenKindTag.IDENTIFIER

    @dataclass
    class Integer:
        value: int
        tag: Literal[TokenKindTag.INTEGER] = TokenKindTag.INTEGER

    @dataclass
    class LeftBrace:
        tag: Literal[TokenKindTag.LEFT_BRACE] = TokenKindTag.LEFT_BRACE

    @dataclass
    class RightBrace:
        tag: Literal[TokenKindTag.RIGHT_BRACE] = TokenKindTag.RIGHT_BRACE

    @dataclass
    class LeftBracket:
        tag: Literal[TokenKindTag.LEFT_BRACKET] = TokenKindTag.LEFT_BRACKET

    @dataclass
    class RightBracket:
        tag: Literal[TokenKindTag.RIGHT_BRACKET] = TokenKindTag.RIGHT_BRACKET

    @dataclass
    class SemiColon:
        tag: Literal[TokenKindTag.SEMICOLON] = TokenKindTag.SEMICOLON

    @dataclass
    class Comma:
        tag: Literal[TokenKindTag.COMMA] = TokenKindTag.COMMA

    @dataclass
    class Equals:
        tag: Literal[TokenKindTag.EQUALS] = TokenKindTag.EQUALS


TokenKind = (
    TokenKindVariant.Keyword
    | TokenKindVariant.Identifier
    | TokenKindVariant.Integer
    | TokenKindVariant.LeftBrace
    | TokenKindVariant.RightBrace
    | TokenKindVariant.LeftBracket
    | TokenKindVariant.RightBracket
    | TokenKindVariant.SemiColon
    | TokenKindVariant.Comma
    | TokenKindVariant.Equals
)


@dataclass
class Span:
    """
    Represents start and end bounds of a token within source.
    """

    start: int = 0
    end: int = 0

    def __add__(self, other: "Span") -> "Span":
        if not isinstance(other, Span):  # pyright: reportUnnecessaryIsInstance=false
            return NotImplemented  # type: ignore[unreachable]

        return Span(self.start, other.end)


@dataclass
class Token:
    """
    Represents token as its type and location in source.
    """

    kind: TokenKind
    span: Span


class LexError(ValueError):
    """
    Base Exception type for all exceptions used by PISS lex tools.
    """

    ...


class UnexpectedEOF(LexError):
    """
    Signifies that the end of input was unexpectedly reached while lexing.
    """

    ...


class UnexpectedCharacter(LexError):
    """
    Signifies that an unexpected character (probably illegal)
    was encountered while lexing.
    """

    ...


class NoMatch(LexError):
    """
    Signifies that the scanned input did not contain an expected lexeme.
    """

    ...


def take_while(data: str, pred: Callable[[str], bool]) -> tuple[str, int]:
    """
    Consume provided input character by character while predicate remains True.

    Parameters:
        data: str - Input to be consumed.
        pred: (str) -> bool - Predicate to check characters against.

    Returns:
        remaining, count = take_while(str, (str) -> bool)
        remaining: str - Unconsumed input.
        count: int - Number of characters consumed.

    Raises:
        NoMatch - Input was consumed with no characters matching predicate.
    """

    current_index = 0

    for char in data:
        should_not_continue = not pred(char)

        if should_not_continue:
            break

        current_index += 1

    if current_index == 0:
        raise NoMatch

    return data[0:current_index], current_index


def skip_until(data: str, pattern: str) -> str:
    """
    Consume provided input character by character until pattern is found.

    Parameters:
        data: str - Input to be consumed.
        pattern: str - Pattern to search for.

    Returns:
        str - The remainder of the input.
    """

    while len(data) and not data.startswith(pattern):
        data = data[1:]

    return data[len(pattern) :]


def tokenize_identifier_or_keyword(
    data: str,
) -> tuple[TokenKindVariant.Identifier | TokenKindVariant.Keyword, int]:
    """
    Attempt to consume Identifier or Keyword token type from input.

    Parameters:
        data: str - Input to tokenize.

    Returns:
        kind, length = tokenize_identifier_or_keyword(str)
        kind: Identifier | Keyword - Type of Token successfully lexed.
        length: int - Number of characters consumed from input.

    Raises:
        UnexpectedEOF - Input was empty.
        UnexpectedCharacter - Input does not start with digit.
        NoMatch - Input did not begin with Identifier or Keyword.
    """

    try:
        first_character = data[0]
    except IndexError:
        raise UnexpectedEOF

    if first_character.isdigit():
        raise UnexpectedCharacter

    name, chars_read = take_while(data, lambda char: char == "_" or char.isalnum())

    if name in [variant.value for variant in KeywordKind]:
        return TokenKindVariant.Keyword(KeywordKind(name)), chars_read

    return TokenKindVariant.Identifier(name), chars_read


def tokenize_integer(data: str) -> tuple[TokenKindVariant.Integer, int]:
    """
    Attempt to consume an Integer token type from input.

    Parameters:
        data: str - Input to tokenize.

    Returns:
        Integer, int - Consumed Integer token type and its length.

    Raises:
        NoMatch - Input did not begin with Integer.
    """

    integer, chars_read = take_while(data, lambda char: char.isdigit())
    return TokenKindVariant.Integer(int(integer)), chars_read


def skip_whitespace(data: str) -> int:
    """
    Consume whitespace input.

    Parameters:
        data: str - Input to consume.

    Returns:
        int - Number of whitespace characters consumed.

    Raises:
        NoMatch - Input did not begin with whitespace.
    """

    try:
        _, chars_read = take_while(data, lambda char: char.isspace())
    except NoMatch:
        return 0

    return chars_read


def skip_comments(data: str) -> int:
    """
    Consume comments input.

    Parameters:
        data: str - Input to consume.

    Returns:
        int - Number of comment characters consumed.
    """

    if data.startswith("//"):
        leftovers = skip_until(data, "\n")
        return len(data) - len(leftovers)

    return 0


def skip(data: str) -> int:
    """
    Consume comments and whitespace from input.

    Parameters:
        data: str - Input to be consumed.

    Returns:
        int - Number of comment and whitespace characters consumed.
    """

    remaining = data

    while True:
        whitespace_count = skip_whitespace(remaining)
        remaining = remaining[whitespace_count:]
        comment_count = skip_comments(remaining)
        remaining = remaining[comment_count:]

        if whitespace_count + comment_count == 0:
            return len(data) - len(remaining)


def token(data: str) -> tuple[TokenKind, int] | None:
    """
    Consume token from input, if it exists.

    Parameters:
        data: str - Input to tokenize.

    Returns:
        None - Input is exhausted.
        TokenKind, int - Kind of consumed token, and its length.

    Raises:
        UnexpectedCharacter - Input contained character which is illegal in all Tokens.
    """

    try:
        next_char = data[0]
    except IndexError:
        return None

    primitive_tokens: dict[str, tuple[TokenKind, int]] = {
        "{": (TokenKindVariant.LeftBrace(), 1),
        "}": (TokenKindVariant.RightBrace(), 1),
        ";": (TokenKindVariant.SemiColon(), 1),
        ",": (TokenKindVariant.Comma(), 1),
        "=": (TokenKindVariant.Equals(), 1),
    }

    token_kind: TokenKind
    length: int

    if next_char in primitive_tokens:
        token_kind, length = primitive_tokens[next_char]

    elif next_char.isdigit():
        token_kind, length = tokenize_integer(data)

    elif next_char == "_" or next_char.isalpha():
        token_kind, length = tokenize_identifier_or_keyword(data)

    else:
        raise UnexpectedCharacter

    return token_kind, length


class Tokenizer:
    """
    Represents an input to be tokenized

    Parameters:
        source: str - The input to be tokenized.

    Attributes:
        current_index: int - Index into source, where lexing is occuring.
        remaining_text: str - Unconsumed portion of source.
    """

    def __init__(self, source: str):
        self.current_index = 0
        self.remaining_text = source

    def __repr__(self) -> str:
        if len(self.remaining_text) < 15:
            return f'Tokenizer("{self.remaining_text}")'
        return f'Tokenizer("{self.remaining_text[0:15]}...")'

    def next_token(self) -> Token | None:
        """
        Consume one token from front of input, if it exists.

        Returns:
            None - Input has been exhausted.
            Token - Consumed Token.
        """

        self.skip_whitespace()

        if not len(self.remaining_text):
            return None

        start = self.current_index

        token_kind_or_none = self._next_token()
        if token_kind_or_none is None:
            return None

        end = self.current_index

        return Token(kind=token_kind_or_none, span=Span(start, end))

    def skip_whitespace(self) -> None:
        """
        Consume whitespace and comments from remaining input.
        """

        skipped = skip(self.remaining_text)
        self.chomp(skipped)

    def _next_token(self) -> TokenKind | None:
        """
        Consume token from remaining input, if it exists.

        Returns:
            None - Input is exhausted.
            TokenKind - Type of Token consumed.

        Raises:
            UnexpectedCharacter - Remaining input contained character which is illegal in all Tokens.
        """

        result = token(self.remaining_text)
        if result is None:
            return None

        token_kind, chars_read = result
        self.chomp(chars_read)

        return token_kind

    def chomp(self, count: int) -> None:
        """
        Remove characters from remaining input.

        Parameters:
            count: int - Number of characters to remove.

        """

        self.remaining_text = self.remaining_text[count:]
        self.current_index += count


def tokenize(data: str) -> list[Token]:
    """
    Produce a list of Tokens from the provided input.

    Parameters:
        data: str - Input to be tokenized.

    Returns:
        list[Tokens] - List of Tokens.
    """

    tokenizer = Tokenizer(data)
    tokens: list[Token] = []

    while True:
        result = tokenizer.next_token()

        if result is None:
            break

        tokens.append(result)

    return tokens
