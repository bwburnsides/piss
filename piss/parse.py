"""
Tools for parsing the PISS grammar.
"""

from dataclasses import dataclass
import enum
from piss import lex
import typing
from typing import Callable
from piss.lex import TokenKind


T = typing.TypeVar("T")


def parse_if(
    parser: Callable[[], T],
    predicate: Callable[[], bool],
) -> T | None:
    """
    Parse T if predicate is True.

    Parameters:
        parser: () -> T - Parser for T.
        predicate: () -> bool - Condition which must be True in order to parse T.

    Returns:
        None - Predicate was not True and T was not parsed.
        T - Parsed T.
    """

    return parser() if predicate() else None


def many(parser: Callable[[], T | None]) -> list[T]:
    """
    Parse and collect Ts.

    Example:
        Consider a parse source that has the following types of items to parse at
        its front: T T T U V T T ...

        Provided parser should return None when U is encountered. Higher order parser will
        return [T, T, T].

    Parameters:
        parser: () -> T | None - Optional T parser. Higher order parsing concludes
        when parser returns None.

    Returns:
        list[T] - Parsed Ts produced via parser.

    Raises:
        RecursionError - Parser did not return None, causing Stack Overflow.
        Any - Uncaught Exception raised by parser.
    """

    items: list[T] = []

    # TODO: more functional way to represent this for sure. maybe something like
    # functools.reduce.
    while True:
        item_or_none = parser()

        if item_or_none is None:
            break

        items.append(item_or_none)

    return items


@dataclass
class Node:
    """
    Represents a generic node in a PISS AST.
    """

    span: lex.Span


@dataclass
class Keyword(Node):
    kind: lex.KeywordKind


@dataclass
class Integer(Node):
    value: int


@dataclass
class Identifier(Node):
    name: str


@dataclass
class Expression(Node):
    expr: Identifier | Integer


@dataclass
class PrimitiveKind(enum.Enum):
    """
    PrimitiveKind enumerates the primitive (builtin) types in PISS grammar.
    These are represented by Keyword tokens.
    """

    Uint = enum.auto()
    Int = enum.auto()


@dataclass
class Type(Node):
    ...


@dataclass
class PrimitiveType(Type):
    type: PrimitiveKind


@dataclass
class IdentifierType(Type):
    type: Identifier


@dataclass
class ArrayType(Type):
    type: Type
    length: Expression


@dataclass
class Field(Node):
    kind: Type
    ident: Identifier


@dataclass
class Definition(Node):
    ...


@dataclass
class Const(Definition):
    kind: Identifier | Type
    ident: Identifier
    expr: Expression


@dataclass
class Struct(Definition):
    ident: Identifier
    fields: list[Field]


@dataclass
class Enum(Definition):
    ident: Identifier
    variants: list[Identifier]


@dataclass
class Typedef(Definition):
    kind: Type
    ident: Identifier


@dataclass
class Module(Definition):
    ident: Identifier
    defs: list["Definition"]


GenericNodeT = typing.TypeVar("GenericNodeT", bound=Node)
GenericNodeU = typing.TypeVar("GenericNodeU", bound=Node)


class ParseError(ValueError):
    """
    Base Exception type for all exceptions used by PISS parse tools.
    """

    ...


class UnexpectedEOF(ParseError):
    """
    Signifies that the end of input was unexpectedly reached while parsing.
    """

    ...


class UnexpectedToken(ParseError):
    """
    Signifies that an unexpected character (probably illegal)
    was encountered while lexing.
    """

    ...


class Parser:
    """
    Owns a stream of tokens and parser combinators to operate on it.

    Parameters:
        tokens: list[Token] - Stream of tokens to parse.
    """

    def __init__(self, tokens: list[lex.Token]):
        self.tokens = tokens
        self.index = 0
        self.state: list[int] = []

    def peek(self) -> TokenKind | None:
        """
        Return type of current token without advancing stream pointer.

        Returns:
            None - Stream is exhausted.
            TokenKind - Kind of token at front of stream.
        """

        try:
            current_token = self.tokens[self.index]
        except IndexError:
            return None

        return current_token.kind

    def next(self) -> lex.Token:
        """
        Return current token and advance the stream pointer.

        Return:
            Token - Token at front of stream.

        Raises:
            UnexpectedEOF - Stream was exhausted.
        """

        try:
            current_token = self.tokens[self.index]
        except IndexError:
            raise UnexpectedEOF

        self.index += 1

        return current_token

    def push(self) -> None:
        """
        Store current state to stack.
        """

        self.state.append(self.index)

    def pop(self) -> None:
        """
        Restore current state from stack.
        """

        self.index = self.drop()

    def drop(self) -> int:
        """
        Drop current state from stack.
        """

        try:
            return self.state.pop()
        except IndexError:
            raise ParseError

    def try_parse(self, parser: Callable[[], T]) -> T | None:
        """
        Attempt to parse T from token stream. No tokens are consumed if parsing fails.

        Parameters:
            parser: () -> T - Parser for T.

        Returns:
            None - T could not be parsed from stream.
            T - Successfully parsed T.

        Raises:
            UnexpectedEOF - Token Stream was exhausted.
            Any - Uncaught Exception raised by parser. UnexpectedToken is handled.
        """

        self.push()
        try:
            return parser()
        except UnexpectedEOF:
            self.pop()
            raise UnexpectedEOF
        except UnexpectedToken:
            self.drop()
            return None

    # TODO: refactor TokenKind relationships to remove cheesy typing.Type needs.
    def and_then(
        self,
        parser: Callable[[], GenericNodeT],
        terminator: typing.Type[TokenKind],
    ) -> GenericNodeT | None:
        """
        Attempt to parse Node from token stream. If successful, parse terminator from token stream. If successful,
        return parsed node.

        Examples:
            Consider and_then(parses_T, +) applied to stream: T + U * V - W ...
            In which case parsed T is returned and T + are consumed.

            Consider and_then(parsed_T, +) applied to stream: T - V + X ...
            In which case UnexpectedToken is raised no tokens are consumed.

            Consider and_then(Parsed_T, +) applied to stream: U + V - X ...
            In which case None is returned and no tokens are consumed.

        Parameters:
            parser: () -> GenericNode - Parser with defines valid GenericNodes.
            terminator: Type[TokenKind] - Type of token which must follow parsed GenericNode in stream.

        Returns:
            GenericNode - Successfully parsed node followed by terminator.
            None - Node could not be parsed from token stream.

        Raises:
            UnexpectedEOF - Token stream was exhausted.
            UnexpectedToken - Succeeding token in stream's type did not match provided terminator type.
            Any - Uncaught Exception raised by parser.
        """

        node_or_none = self.try_parse(parser)

        if node_or_none is not None:
            self.parse_token(terminator)

        return node_or_none

    def either_or(
        self,
        first_choice: Callable[[], GenericNodeT],
        second_choice: Callable[[], GenericNodeU],
    ) -> GenericNodeT | GenericNodeU:
        """
        Parse one of two choices from parse source. Attempt to parse first choice and return item if successful.
        Otherwise, parse second choice and return item.

        Examples:
            Consider either_or(parses_T, parses_U) applied to stream: T ...
            In which case parsed T is returned and T is consumed.

            Consider either_or(parses_T, parses_U) applied to stream: U ...
            In which case parsed U is returned and U is consumed.

            Consider either_or(parses_T, parses_U) applied to stream: V ...
            Higher order parser fails.

        Parameters:
            first_choice: () -> NodeT - Parser for higher preference choice.
            second_choice: () -> NodeU - Parser for lower preference choice.

        Returns:
            NodeT - Successfully parsed Node from first choice parser.
            NodeU - Successfully parsed Node from second choice parser.

        Raises:
            UnexpectedToken - Lower preference NodeU could not be parsed from stream. (Implies that NodeT parsing failed too.)
            Any - Uncaught Exception raised by either parser. UnexpectedToken thrown by higher preference NodeT parser is handled.
        """

        choices: list[Callable[[], GenericNodeT | GenericNodeU]] = [
            first_choice,
            second_choice,
        ]
        return self.choice(choices)

    def between(
        self,
        before: typing.Type[TokenKind],
        after: typing.Type[TokenKind],
        parser: Callable[[], T],
    ) -> tuple[lex.Token, lex.Token, T]:
        """
        Parse NodeT with preceding and succeeding tokens of specified kinds from token stream.

        Example:
            Consider a token stream with the following tokens at its front: LeftBrace T RightBrace ...

            Higher order parser will succeed if specified before and after TokenKinds are LeftBrace
            and RightBrace respectively.

        Parameters:
            before: Type[TokenKind] - TokenKind that must be matched before T is parsed.
            after: Type[TokenKind] - TokenKind that must be matched after T is parsed.
            parser: () -> T - Parser which defines valid Ts.

        Returns:
            lex.Token, lex.Token, T - Consumed before token, after token, and parsed T, respectively.

        Raises:
            UnexpectedToken - Stream could not parse the before token, T, or after token.
            UnexpectedEOF - Stream was exhausted while parsing the before token, T, or after Token.
            Any - Uncaught Exception raised by parser.
        """

        before_token = self.parse_token(before)
        node = parser()
        after_token = self.parse_token(after)

        return before_token, after_token, node

    def choice(self, parsers: list[Callable[[], T]]) -> T:
        for parser in parsers:
            node_or_none = self.try_parse(parser)

            if node_or_none is not None:
                return node_or_none

        raise UnexpectedToken

    # TODO
    def parse_token(self, kind: typing.Type[TokenKind]) -> lex.Token:
        """
        Parse token of given kind from token stream.

        Parameters:
            kind: TokenKindTag - Type of token to parse.

        Returns:
            Token - Parsed token.

        Raises:
            UnexpectedEOF - Stream was exhausted.
            UnexpectedToken - Token was not expected type.
        """

        peeked = self.peek()

        if peeked is None:
            raise UnexpectedEOF

        if not isinstance(peeked, kind):  # pyright: reportUnnecessaryIsInstance=false
            raise UnexpectedToken

        return self.next()

    # TODO
    def parse_keyword(self, kind: lex.KeywordKind) -> Keyword:
        """
        Parse Keyword given kind from token stream.

        Parameters:
            kind: KeywordKind - Kind of Keyword to parse.

        Returns:
            Keyword - Parsed Keyword.

        Raises:
            UnexpectedEOF - Stream was exhausted.
            UnexpectedToken - Token was not expected type.
        """

        self.push()
        token = self.parse_token(lex.Keyword)
        if not isinstance(token.kind, lex.Keyword):
            self.pop()
            raise UnexpectedToken

        if token.kind.keyword is not kind:
            self.pop()
            raise UnexpectedToken

        self.drop()
        return Keyword(token.span, token.kind.keyword)

    # TODO
    def parse_identifier(self) -> Identifier:
        """
        Parse Identifier from token stream.

        Returns:
            Identifier - Parsed Identifier.

        Raises:
            UnexpectedEOF - Stream was exhausted.
            UnexpectedToken - Token was not Identifier.
        """

        token = self.parse_token(lex.Identifier)
        if not isinstance(token.kind, lex.Identifier):
            raise UnexpectedToken

        return Identifier(token.span, token.kind.name)

    # TODO
    def parse_integer(self) -> Integer:
        """
        Parse Integer from token stream.

        Returns:
            Integer - Parsed Integer.

        Raises:
            UnexpectedEOF - Stream was exhausted.
            UnexpectedToken - Token was not Integer.
        """

        token = self.parse_token(lex.Integer)
        if not isinstance(token.kind, lex.Integer):
            raise UnexpectedToken

        return Integer(token.span, token.kind.value)

    def parse_expression(self) -> Expression:
        """
        Parse Expression from token stream.

        Expression = Identifier | Integer

        Returns:
            Expression - Parsed Expression.

        Raises:
            UnexpectedEOF - Stream was exhausted.
            UnexpectedToken - Tokens could not parse into Expression.
        """

        ident_or_int = self.either_or(self.parse_identifier, self.parse_integer)
        return Expression(ident_or_int.span, ident_or_int)

    def parse_primitive_type(self) -> PrimitiveType:
        primitives_mapping: dict[lex.KeywordKind, PrimitiveKind] = {
            lex.KeywordKind.Uint: PrimitiveKind.Uint,
            lex.KeywordKind.Int: PrimitiveKind.Int,
        }

        parsed_keyword = self.choice(
            [lambda: self.parse_keyword(kw) for kw in primitives_mapping]
        )

        return PrimitiveType(
            parsed_keyword.span, primitives_mapping[parsed_keyword.kind]
        )

    def parse_identifier_type(self) -> IdentifierType:
        ident = self.parse_identifier()
        return IdentifierType(ident.span, ident)

    def parse_type(self) -> Type:
        """
        Parse Type from token stream.

        Type = PrimitiveType (LeftBracket Expression RightBracket)*
             | Identifier (LeftBracket Expression RightBracket)*

        Returns:
            Type - Parsed Type.

        Raises:
            UnexpectedEOF - Stream was exhausted.
            UnexpectedToken - Tokens could not parse into Type.
        """

        parsed_type: Type = self.either_or(
            self.parse_primitive_type,
            self.parse_identifier_type,
        )

        def maybe_parse_bounds() -> tuple[Expression, lex.Span] | None:
            def parse_bounds() -> tuple[Expression, lex.Span]:
                _, right_bracket, expr = self.between(
                    lex.LeftBracket, lex.RightBracket, self.parse_expression
                )

                return expr, right_bracket.span

            return parse_if(
                parse_bounds, lambda: isinstance(self.peek(), lex.LeftBracket)
            )

        exprs_and_spans = many(maybe_parse_bounds)

        # TODO: there is a functional way to express this. something similar
        # to functools.reduce maybe.
        for expr, span in exprs_and_spans:
            start_span = parsed_type.span

            parsed_type = ArrayType(
                start_span + span,
                parsed_type,
                expr,
            )

        return parsed_type

    def parse_field(self) -> Field:
        """
        Parse Field from token stream.

        Field = Type Identifier

        Returns:
            Field - Parsed Field.

        Raises:
            UnexpectedEOF - Stream was exhausted.
            UnexpectedToken - Tokens could not parse into Field.
        """

        type = self.parse_type()
        ident = self.parse_identifier()

        return Field(type.span + ident.span, type, ident)

    def parse_const(self) -> Const:
        """
        Parse Const from token stream.

        ConstDefinition = Keyword::CONST Field Equals Expression SemiColon

        Returns:
            Const - Parsed Const.

        Raises:
            UnexpectedEOF - Stream was exhausted.
            UnexpectedToken - Tokens could not parse into Const.
        """

        const = self.parse_keyword(lex.KeywordKind.Const)
        field = self.parse_field()

        self.parse_token(lex.Equals)

        expr = self.parse_expression()

        semicolon = self.parse_token(lex.SemiColon)

        return Const(const.span + semicolon.span, field.kind, field.ident, expr)

    def parse_struct(self) -> Struct:
        """
        Parse Struct from token stream.

        StructDefinition = Keyword::STRUCT Identifier LeftBrace (Field Comma)* RightBrace SemiColon

        Returns:
            Struct - Parsed Struct.

        Raises:
            UnexpectedEOF - Stream was exhausted.
            UnexpectedToken - Tokens could not parse into Struct.
        """

        struct = self.parse_keyword(lex.KeywordKind.Struct)
        ident = self.parse_identifier()

        _, _, fields = self.between(
            lex.LeftBrace,
            lex.RightBrace,
            lambda: many(lambda: self.and_then(self.parse_field, lex.Comma)),
        )

        semicolon = self.parse_token(lex.SemiColon)

        return Struct(struct.span + semicolon.span, ident, fields)

    def parse_enum(self) -> Enum:
        """
        Parse Enum from token stream.

        EnumDefinition = Keyword::ENUM Identifier LeftBrace (Identifier Comma)* RightBrace SemiColon

        Returns:
            Enum - Parsed Enum.

        Raises:
            UnexpectedEOF - Stream was exhausted.
            UnexpectedToken - Tokens could not parse into Enum.
        """

        enum = self.parse_keyword(lex.KeywordKind.Enum)
        ident = self.parse_identifier()

        _, _, variants = self.between(
            lex.LeftBrace,
            lex.RightBrace,
            lambda: many(lambda: self.and_then(self.parse_identifier, lex.Comma)),
        )

        semicolon = self.parse_token(lex.SemiColon)

        return Enum(enum.span + semicolon.span, ident, variants)

    def parse_typedef(self) -> Typedef:
        """
        Parse Typedef from token stream.

        TypedefDefinition = Keyword::TYPEDEF Field SemiColon

        Returns:
            Typedef - Parsed Typedef.

        Raises:
            UnexpectedEOF - Stream was exhausted.
            UnexpectedToken - Tokens could not parse into Typedef.
        """

        typedef = self.parse_keyword(lex.KeywordKind.Typedef)
        field = self.parse_field()

        semicolon = self.parse_token(lex.SemiColon)

        return Typedef(typedef.span + semicolon.span, field.kind, field.ident)

    def parse_definition(self) -> Definition:
        """
        Parse Definition from token stream.

        Definition = ConstDefinition
                | StructDefinition
                | EnumDefinition
                | TypedefDefinition

        Returns:
            Definition - Parsed Definition

        Raises:
            UnexpectedEOF - Stream was exhausted.
            UnexpectedToken - Tokens could not parse into Definition.
        """

        return self.choice(
            [
                self.parse_const,
                self.parse_struct,
                self.parse_enum,
                self.parse_typedef,
            ]
        )

    def parse_module(self) -> Module:
        """
        Parse Definition from token stream.

        Module = Keyword::MODULE Identifier LeftBrace Definition* RightBrace SemiColon

        Returns:
            Module - Parsed Module

        Raises:
            UnexpectedEOF - Stream was exhausted.
            UnexpectedToken - Tokens could not parse into Module.
        """

        module = self.parse_keyword(lex.KeywordKind.Module)
        ident = self.parse_identifier()

        _, _, definitions = self.between(
            before=lex.LeftBrace,
            after=lex.RightBrace,
            parser=lambda: many(lambda: self.try_parse(self.parse_definition)),
        )

        semicolon = self.parse_token(lex.SemiColon)

        return Module(module.span + semicolon.span, ident, definitions)
