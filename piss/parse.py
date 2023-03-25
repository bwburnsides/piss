"""
Tools for parsing the PISS grammar.
"""

from dataclasses import dataclass
import enum
from piss import lex
import typing
from typing import Callable
from piss.lex import TokenKind


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

    UINT = enum.auto()
    INT = enum.auto()


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

    def peek(self) -> lex.TokenKind | None:
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

        # Attempt to parse an Identifier. If there is an UnexpectedToken, then we could
        # not successfully parse one, and now we'll try to parse an Integer.
        self.push()
        try:
            ident = self.parse_identifier()
        except UnexpectedToken:
            self.pop()
        else:
            self.drop()
            return Expression(ident.span, ident)

        # Don't try to catch UnexpectedToken here. If its raised that
        # means that neither an Identifier or an Integer could be parsed.
        # This means that no Expression could be parsed since we've
        # already tried to parse an Identifier.
        integer = self.parse_integer()
        return Expression(integer.span, integer)

    def parse_type(self) -> Type:
        """
        Parse Type from token stream.

        Type = PrimitiveType
              | Identifier
              | Type LeftBracket Expression RightBracket

        Returns:
            Type - Parsed Type.

        Raises:
            UnexpectedEOF - Stream was exhausted.
            UnexpectedToken - Tokens could not parse into Type.
        """

        primitives = [
            (lex.KeywordKind.UINT, PrimitiveKind.UINT),
            (lex.KeywordKind.INT, PrimitiveKind.INT),
        ]

        parsed_type: Type | None = None

        # Search for a Keyword token corresponding to the PrimitiveType
        # KeywordKind. If there is an UnexpectedToken, then we'll try to parse
        # the next PrimitiveType KeywordKind. If there isn't, then we've parsed
        # the base of a Type and can progress to checking for array types.
        for keyword, primitive in primitives:
            self.push()
            try:
                parsed_primitive_type_keyword = self.parse_keyword(keyword)
            except UnexpectedToken:
                self.pop()
                continue
            else:
                self.drop()
                parsed_type = PrimitiveType(
                    parsed_primitive_type_keyword.span, primitive
                )

        # If no PrimitiveType could be parsed, then the Type base must be an Identifier.
        # This Identifier corresponds to the name of a user-defined type, maybe via
        # a typedef, struct, or enum definition. Don't try to catch an exceptions
        # since this is the last possibility for a valid Type base.
        if parsed_type is None:
            ident = self.parse_identifier()
            parsed_type = IdentifierType(ident.span, ident)

        start_span = parsed_type.span

        # Now we check if this type is an array type.
        while True:
            # Loop with the following pattern: look for left bracket, then look for expression, then look for right bracket.
            # If no left bracket, we're done (break and return)
            # If anything after left bracket is missing, then UnexpectedToken / UnexpectedEOF

            self.push()
            try:
                self.parse_token(lex.LeftBracket)
            except ParseError:
                self.pop()
                break

            self.drop()

            expr = self.parse_expression()
            left_bracket = self.parse_token(lex.RightBracket)

            end_span = left_bracket.span

            parsed_type = ArrayType(
                start_span + end_span,
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

        const = self.parse_keyword(lex.KeywordKind.CONST)

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

        struct = self.parse_keyword(lex.KeywordKind.STRUCT)

        ident = self.parse_identifier()

        self.parse_token(lex.LeftBrace)

        # An enum can contain any number of fields in it, which means we need to loop in order to parse
        # them all. A variant is defined as an Identifier that must be followed by a Comma.
        fields: list[Field] = []
        while True:
            # Attempt to parse a Field. If there is an UnexpectedToken, then there is not a field at the
            # front of the Token stream and we are done parsing fields and can break.
            try:
                self.push()
                field = self.parse_field()
            except UnexpectedToken:
                self.pop()
                break
            else:
                self.drop()
                fields.append(field)

            # If we successfully parsed an Identifier then it must be followed by a Comma, do don't catch any
            # exceptions which may occur.
            self.parse_token(lex.Comma)

        self.parse_token(lex.RightBrace)
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

        enum = self.parse_keyword(lex.KeywordKind.ENUM)

        ident = self.parse_identifier()

        self.parse_token(lex.LeftBrace)

        # An enum can contain any number of variants in it, which means we need to loop in order to parse
        # them all. A variant is defined as an Identifier that must be followed by a Comma.
        variants: list[Identifier] = []
        while True:
            # Attempt to parse an Identifier, which names the variant. If there is an UnexpectedToken, then
            # there is not a variant at the front of the Token stream and we are done parsing variants and can break.
            self.push()
            try:
                variant = self.parse_identifier()
            except UnexpectedToken:
                self.pop()
                break
            else:
                self.drop()
                variants.append(variant)

            # If we successfully parsed an Identifier then it must be followed by a Comma, so don't catch any
            # exceptions which may occur.
            self.parse_token(lex.Comma)

        self.parse_token(lex.RightBrace)
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

        typedef = self.parse_keyword(lex.KeywordKind.TYPEDEF)

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

        definition_parsers: list[Callable[[], Definition]] = [
            self.parse_const,
            self.parse_struct,
            self.parse_enum,
            self.parse_typedef,
        ]

        # Try to parse each type of definition. If it succeeds, then return that definition.
        # However, a ParseError might be raised. These include UnexpectedEOF and UnexpectedToken.
        # In the case of an UnexpectedEOF, that means that the token stream was unexpectedly
        # exhausted. This is a syntactic error, it means that tokens were missing. Maybe a missing
        # semicolon, so we raise that exception to our caller. However, if for example an
        # UnexpectedToken exception occurs, this likely means that the token stream doesn't
        # currently have the target definition at the front. In that case, we ignore the
        # exception and continue onwards and attempt to parse the next type of definition.

        for parser in definition_parsers:
            self.push()
            try:
                return parser()
            except UnexpectedEOF:
                self.pop()
                raise UnexpectedEOF
            except UnexpectedToken:
                self.drop()
                continue

        raise UnexpectedToken

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

        module = self.parse_keyword(lex.KeywordKind.MODULE)

        ident = self.parse_identifier()

        self.parse_token(lex.LeftBrace)

        definitions: list[Definition] = []

        while True:
            try:
                self.push()
                definition = self.parse_definition()
            except UnexpectedToken:
                self.pop()
                break
            except ParseError:
                self.pop()
                raise ParseError
            else:
                self.drop()
                definitions.append(definition)

        self.parse_token(lex.RightBrace)
        semicolon = self.parse_token(lex.SemiColon)

        return Module(module.span + semicolon.span, ident, definitions)
