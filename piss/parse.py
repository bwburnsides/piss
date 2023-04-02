"""
Tools for parsing the PISS grammar.
"""

from dataclasses import dataclass
import enum
import typing
from typing import Callable
from abc import ABC, abstractmethod
from piss import lex
from piss.lex import TokenKind
from functools import partial

T = typing.TypeVar("T")
U = typing.TypeVar("U")

ParserType = Callable[[], T]
OptionalParserType = Callable[[], T | None]


def parse_if(
    parser: ParserType[T],
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


def many(parser: OptionalParserType[T]) -> list[T]:
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

    while True:
        item_or_none = parser()

        if item_or_none is None:
            break

        items.append(item_or_none)

    return items


def vector(parser: OptionalParserType[T], delimiter: OptionalParserType[U]) -> list[T]:
    items: list[T] = []

    while True:
        item_or_none = parser()
        if item_or_none is None:
            break

        items.append(item_or_none)

        delimiter_or_none = delimiter()
        if delimiter_or_none is None:
            break

    return items


@dataclass
class Node(ABC):
    """
    Represents a generic node in a PISS AST.
    """

    span: lex.Span

    @abstractmethod
    def accept(self, visitor: "NodeVisitor") -> None:
        raise NotImplementedError


@dataclass
class Keyword(Node):
    kind: lex.KeywordKind

    def accept(self, visitor: "NodeVisitor") -> None:
        visitor.visit_keyword(self)


@dataclass
class Integer(Node):
    value: int

    def accept(self, visitor: "NodeVisitor") -> None:
        visitor.visit_integer(self)


@dataclass
class Identifier(Node):
    name: str

    def accept(self, visitor: "NodeVisitor") -> None:
        visitor.visit_identifier(self)


@dataclass
class Expression(Node):
    expr: Identifier | Integer

    def accept(self, visitor: "NodeVisitor") -> None:
        visitor.visit_expression(self)


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

    def accept(self, visitor: "NodeVisitor") -> None:
        visitor.visit_primitive_type(self)


@dataclass
class IdentifierType(Type):
    type: Identifier

    def accept(self, visitor: "NodeVisitor") -> None:
        visitor.visit_identifier_type(self)


@dataclass
class ArrayType(Type):
    type: Type
    length: Expression

    def accept(self, visitor: "NodeVisitor") -> None:
        visitor.visit_array_type(self)


@dataclass
class Field(Node):
    kind: Type
    ident: Identifier

    def accept(self, visitor: "NodeVisitor") -> None:
        visitor.visit_field(self)


@dataclass
class Definition(Node):
    ...


@dataclass
class Const(Definition):
    kind: Identifier | Type
    ident: Identifier
    expr: Expression

    def accept(self, visitor: "NodeVisitor") -> None:
        visitor.visit_const(self)


@dataclass
class Struct(Definition):
    ident: Identifier
    fields: list[Field]

    def accept(self, visitor: "NodeVisitor") -> None:
        visitor.visit_struct(self)


@dataclass
class Enum(Definition):
    ident: Identifier
    variants: list[Identifier]

    def accept(self, visitor: "NodeVisitor") -> None:
        visitor.visit_enum(self)


@dataclass
class Typedef(Definition):
    kind: Type
    ident: Identifier

    def accept(self, visitor: "NodeVisitor") -> None:
        visitor.visit_typedef(self)


@dataclass
class Module(Definition):
    ident: Identifier
    definitions: list["Definition"]

    def accept(self, visitor: "NodeVisitor") -> None:
        for definition in self.definitions:
            definition.accept(visitor)


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

    def __init__(self, tokens: list[lex.Token[TokenKind]]):
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

    def next(self) -> lex.Token[TokenKind]:
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

    def try_parse(self, parser: ParserType[T]) -> T | None:
        """
        Attempt to parse T from token stream. No tokens are consumed
        if unexpected token is encountered.

        Parameters:
            parser: () -> T - Parser for T.

        Returns:
            None - T could not be parsed from stream.
            T - Successfully parsed T.

        Raises:
            UnexpectedEOF - Token Stream was exhausted.
            Any - Uncaught Exception raised by parser. UnexpectedToken is handled.
        """

        clean_up: Callable[[], int] | Callable[[], None] = self.drop

        self.push()

        try:
            item = parser()
        except UnexpectedEOF:
            clean_up = self.pop
            raise UnexpectedEOF
        except UnexpectedToken:
            clean_up = self.pop
            return None
        else:
            return item
        finally:
            clean_up()

    def then_check(
        self,
        parser: ParserType[T],
        check: Callable[[T], bool],
    ) -> T:
        """
        Parse node and then verify post condition. Parsed node is only returned if post condition is True.
        Tokens are not consumed if condition is not True.

        Parameters:
            parser: () -> T - Parser for node type T.

        Returns:
            T - Parsed node which fulfilled post check.

        Raises:
            UnexpectedToken - Stream could not parse the before token, T, or after token.
            UnexpectedEOF - Stream was exhausted while parsing the before token, T, or after Token.
        """

        item_or_none = self.try_parse(parser)

        if item_or_none is not None and check(item_or_none):
            return item_or_none

        raise UnexpectedToken

    def and_then(
        self,
        parser: ParserType[GenericNodeT],
        terminator: typing.Type[TokenKind],
    ) -> GenericNodeT | None:
        """
        Attempt to parse Node from token stream. If successful, parse terminator from token stream. If successful,
        return parsed node.

        Examples:
            Consider and_then(parses_T, +) applied to stream: T + U * V - W ...
            In which case parsed T is returned and T + are consumed.

            Consider and_then(parsed_T, +) applied to stream: T - V + X ...
            In which case UnexpectedToken is raised and no tokens are consumed.

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

        def check_terminator(_: GenericNodeT) -> bool:
            token_or_none = self.try_parse(lambda: self.parse_token(terminator))

            if token_or_none is None:
                return False

            return True

        return self.try_parse(lambda: self.then_check(parser, check=check_terminator))

    def either_or(
        self,
        first_choice: ParserType[GenericNodeT],
        second_choice: ParserType[GenericNodeU],
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

        # either_or is really just a special case of the choice parser. Each are given a set of parsers and will return the result
        # of whichever one succeeds first, and will raise an Exception if parsing fails. In the case of either_or, the set of parsers
        # to choose from is strictly two. We implement either_or by constructing a list from the parser parameters, then passing it
        # to choice to take care of.
        choices: list[ParserType[GenericNodeT | GenericNodeU]] = [
            first_choice,
            second_choice,
        ]
        return self.choice(choices)

    def between(
        self,
        before: typing.Type[lex.GenericTokenKindT],
        after: typing.Type[lex.GenericTokenKindU],
        parser: ParserType[T],
    ) -> tuple[lex.Token[lex.GenericTokenKindT], lex.Token[lex.GenericTokenKindU], T]:
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

    def choice(self, parsers: list[ParserType[T]]) -> T:
        """
        Parse node from a choice of parsers. Parsers are chosen sequentially from the start
        of the list until the first success.

        Parameters:
            parsers: list[() -> T] - List of parsers to choose from.

        Returns:
            T - First successfully parsed T node.

        Raises:
            UnexpectedToken - Failed to choose any parser due to unexpected tokens in stream.
            Any - Uncaught Exception raised by any parser. UnexpectedToken is handled.
        """

        for parser in parsers:
            node_or_none = self.try_parse(parser)

            if node_or_none is not None:
                return node_or_none

        raise UnexpectedToken

    def parse_token(
        self, kind: typing.Type[lex.GenericTokenKindT]
    ) -> lex.Token[lex.GenericTokenKindT]:
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

        if not isinstance(peeked, kind):
            raise UnexpectedToken(f"Got {peeked} expected {kind}")

        return typing.cast(lex.Token[lex.GenericTokenKindT], self.next())

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

        token: lex.Token[lex.Keyword] = self.then_check(
            parser=lambda: self.parse_token(lex.Keyword),
            check=lambda token: token.kind.keyword is kind,
        )

        return Keyword(token.span, token.kind.keyword)

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
        return Identifier(token.span, token.kind.name)

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

        # There is some idiotic lexical scoping behavior in Python that precludes me from using the
        # same lambda syntax for creating this list of functions as I have else where in the parser.
        # To get around the weird behavior, I am using functools.partial in order to bind the `kw` argument
        # for parse_keyword in place of lambda. It took most of a day to figure out WTF was going on here.
        #
        # To any readers - be wary of every creating lambdas in a loop that are meant to be called
        # later.
        #
        # Read the explanation in the Python docs here:
        # https://docs.python.org/3.4/faq/programming.html
        # #why-do-lambdas-defined-in-a-loop-with-different-values-all-return-the-same-result
        parsed_keyword = self.choice(
            [partial(self.parse_keyword, kind=kw) for kw in primitives_mapping]
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

        def parse_bounds() -> tuple[Expression, lex.Span]:
            _before, right_bracket, expr = self.between(
                lex.LeftBracket, lex.RightBracket, self.parse_expression
            )

            return expr, right_bracket.span

        exprs_and_spans = many(
            lambda: parse_if(
                parse_bounds, lambda: isinstance(self.peek(), lex.LeftBracket)
            )
        )

        start_span = parsed_type.span

        for expr, span in exprs_and_spans:
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

        _before, _after, fields = self.between(
            lex.LeftBrace,
            lex.RightBrace,
            lambda: vector(
                lambda: self.try_parse(self.parse_field),
                lambda: self.try_parse(lambda: self.parse_token(lex.Comma)),
            ),
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

        _before, _after, variants = self.between(
            lex.LeftBrace,
            lex.RightBrace,
            lambda: vector(
                lambda: self.try_parse(self.parse_identifier),
                lambda: self.try_parse(lambda: self.parse_token(lex.Comma)),
            ),
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

        _before, _after, definitions = self.between(
            before=lex.LeftBrace,
            after=lex.RightBrace,
            parser=lambda: many(lambda: self.try_parse(self.parse_definition)),
        )

        semicolon = self.parse_token(lex.SemiColon)

        return Module(module.span + semicolon.span, ident, definitions)


class NodeVisitor(ABC):
    @abstractmethod
    def visit_keyword(self, keyword: Keyword) -> None:
        raise NotImplementedError

    @abstractmethod
    def visit_integer(self, integer: Integer) -> None:
        raise NotImplementedError

    @abstractmethod
    def visit_identifier(self, ident: Identifier) -> None:
        raise NotImplementedError

    @abstractmethod
    def visit_expression(self, expr: Expression) -> None:
        raise NotImplementedError

    @abstractmethod
    def visit_primitive_type(self, type: PrimitiveType) -> None:
        raise NotImplementedError

    @abstractmethod
    def visit_identifier_type(self, type: IdentifierType) -> None:
        raise NotImplementedError

    @abstractmethod
    def visit_array_type(self, type: ArrayType) -> None:
        raise NotImplementedError

    @abstractmethod
    def visit_field(self, field: Field) -> None:
        raise NotImplementedError

    @abstractmethod
    def visit_typedef(self, typedef: Typedef) -> None:
        raise NotImplementedError

    @abstractmethod
    def visit_const(self, const: Const) -> None:
        raise NotImplementedError

    @abstractmethod
    def visit_enum(self, enum: Enum) -> None:
        raise NotImplementedError

    @abstractmethod
    def visit_struct(self, struct: Struct) -> None:
        raise NotImplementedError

    @abstractmethod
    def visit_module(self, module: Module) -> None:
        raise NotImplementedError
