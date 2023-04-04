from dataclasses import dataclass
import enum
import typing
from pprint import pprint

from piss import lex, node, parse


class SemanticError(ValueError):
    ...


class IdentifierRedefinition(SemanticError):
    ...


class UndefinedIdentifier(SemanticError):
    ...


class SymbolKind(enum.Enum):
    Struct = enum.auto()
    Enum = enum.auto()
    Const = enum.auto()
    Typedef = enum.auto()
    FieldIdentifier = enum.auto()
    VariantIdentifier = enum.auto()


@dataclass
class Symbol:
    kind: SymbolKind
    type: node.Type | None = None
    link: typing.Optional["SymbolTable"] = None


class SymbolTable:
    def __init__(self) -> None:
        self.symbols: dict[str, Symbol] = {}

    def __setitem__(self, name: str, symbol: Symbol) -> None:
        if name in self.symbols:
            raise IdentifierRedefinition

        self.symbols[name] = symbol

    def __getitem__(self, key: str) -> Symbol:
        try:
            return self.symbols[key]
        except KeyError:
            raise UndefinedIdentifier


class SymbolTableVisitor(node.NodeVisitor):
    def __init__(self) -> None:
        self.global_symbols = SymbolTable()
        # self.local_scope_stack: list[SymbolTable] = []

    def visit_module(self, module: node.Module) -> None:
        raise SemanticError

    def visit_struct(self, struct: node.Struct) -> None:
        struct_symbol_table = SymbolTable()

        for field in struct.fields:
            symbol = Symbol(
                kind=SymbolKind.FieldIdentifier,
                type=field.kind,
            )

            struct_symbol_table[field.ident.name] = symbol

        symbol = Symbol(kind=SymbolKind.Struct, type=None, link=struct_symbol_table)

        self.global_symbols[struct.ident.name] = symbol

    def visit_enum(self, enum: node.Enum) -> None:
        enum_symbol_table = SymbolTable()

        for variant in enum.variants:
            symbol = Symbol(
                kind=SymbolKind.VariantIdentifier,
                type=None,
            )

            enum_symbol_table[variant.name] = symbol

        symbol = Symbol(kind=SymbolKind.Enum, type=None, link=enum_symbol_table)

    def visit_const(self, const: node.Const) -> None:
        self.global_symbols[const.ident.name] = Symbol(
            kind=SymbolKind.Const,
            type=const.kind,
        )

    def visit_typedef(self, typedef: node.Typedef) -> None:
        self.global_symbols[typedef.ident.name] = Symbol(
            kind=SymbolKind.Typedef,
            type=typedef.kind,
        )

    def visit_keyword(self, keyword: node.Keyword) -> None:
        raise SemanticError

    def visit_integer(self, integer: node.Integer) -> None:
        raise SemanticError

    def visit_identifier(self, ident: node.Identifier) -> None:
        raise SemanticError

    def visit_expression(self, expr: node.Expression) -> None:
        raise SemanticError

    def visit_primitive_type(self, type: node.PrimitiveType) -> None:
        raise SemanticError

    def visit_identifier_type(self, type: node.IdentifierType) -> None:
        raise SemanticError

    def visit_array_type(self, type: node.ArrayType) -> None:
        raise SemanticError

    def visit_field(self, field: node.Field) -> None:
        raise SemanticError


sample = """
module Foo {
    typedef int MyInt;

    struct Person {
        uint age,
        int debt
    };

    enum Color {
        Red, Green, Blue, Green
    };

    // const int THREE = 3;

    struct Span {
        int start,
        int stop,
    };

    enum TokenKind {
        LeftBrace, RightBrace, SemiColon
    };

    struct Token {
        Span span,
        TokenKind kind
    };

    typedef int[THREE] three_array;
    typedef int[THREE][10] matrix_type;

    struct SomeStruct {
        int Foo,
        int Bar,
        uint Baz
    };
};

// module Bar {};
"""

tokens = lex.tokenize(sample)
modules = parse.parse(tokens)

visitor = SymbolTableVisitor()

modules[0].accept(visitor)
pprint(visitor.global_symbols.symbols)
