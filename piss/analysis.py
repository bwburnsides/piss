from dataclasses import dataclass
import dataclasses
import enum
import typing
from pprint import pprint
from operator import attrgetter

from piss import lex, node, parse


class SemanticError(ValueError):
    ...


class IdentifierRedefinition(SemanticError):
    ...


class UndefinedIdentifier(SemanticError):
    ...


class SymbolKind(enum.Enum):
    Module = enum.auto()
    Struct = enum.auto()
    Enum = enum.auto()
    Const = enum.auto()
    Typedef = enum.auto()
    Identifier = enum.auto()


@dataclass(repr=False)
class Symbol:
    kind: SymbolKind
    type: node.Type | None = None  # TODO: should this be Optional ?
    link: typing.Optional["SymbolTable"] = None

    def __repr__(self) -> str:
        nodef_f_vals = (
            (f.name, attrgetter(f.name)(self))
            for f in dataclasses.fields(self)
            if f.name != "type" and f.name != "link"
        )

        nodef_f_repr = ", ".join(f"{name}={value}" for name, value in nodef_f_vals)
        return f"{self.__class__.__name__}({nodef_f_repr})"


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


class SymbolTableVisitor(node.NodeVisitor[None]):
    def __init__(self) -> None:
        self.table = SymbolTable()

    def visit_module(self, module: node.Module) -> None:
        module_visitor = SymbolTableVisitor()

        for definition in module.definitions:
            definition.accept(module_visitor)

        self.table[module.ident.name] = Symbol(
            kind=SymbolKind.Module,
            link=module_visitor.table,
        )

    def visit_struct(self, struct: node.Struct) -> None:
        struct_visitor = SymbolTableVisitor()

        for field in struct.fields:
            field.accept(struct_visitor)

        self.table[struct.ident.name] = Symbol(
            kind=SymbolKind.Struct, link=struct_visitor.table
        )

    def visit_enum(self, enum: node.Enum) -> None:
        enum_visitor = SymbolTableVisitor()

        for variant in enum.variants:
            variant.accept(enum_visitor)

        self.table[enum.ident.name] = Symbol(
            kind=SymbolKind.Enum, type=None, link=enum_visitor.table
        )

    def visit_const(self, const: node.Const) -> None:
        self.table[const.ident.name] = Symbol(
            kind=SymbolKind.Const,
            type=const.kind,
        )

    def visit_typedef(self, typedef: node.Typedef) -> None:
        self.table[typedef.ident.name] = Symbol(
            kind=SymbolKind.Typedef,
            type=typedef.kind,
        )

    def visit_keyword(self, keyword: node.Keyword) -> None:
        raise SemanticError

    def visit_integer(self, integer: node.Integer) -> None:
        raise SemanticError

    def visit_identifier(self, ident: node.Identifier) -> None:
        self.table[ident.name] = Symbol(kind=SymbolKind.Identifier)

    def visit_expression(self, expr: node.Expression) -> None:
        ...

    def visit_primitive_type(self, type: node.PrimitiveType) -> None:
        ...

    def visit_identifier_type(self, type: node.IdentifierType) -> None:
        ...

    def visit_array_type(self, type: node.ArrayType) -> None:
        ...

    def visit_field(self, field: node.Field) -> None:
        self.table[field.ident.name] = Symbol(
            kind=SymbolKind.Identifier, type=field.kind
        )


sample = """
module Foo {
    typedef int MyInt;

    struct Person {
        uint age,
        int debt
    };

    enum Color {
        Red, Green, Blue,
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


def print_tree(tree: SymbolTable, indent=0):
    for ident, symbol in tree.symbols.items():
        print(indent*" ", end="")
        print(ident, end=": ")
        pprint(symbol)
        if symbol.link is not None:
            print_tree(symbol.link, indent=indent+4)


tokens = lex.tokenize(sample)
modules = parse.parse(tokens)

visitor = SymbolTableVisitor()

# try:
for i, module in enumerate(modules):
    module.accept(visitor)

# print_tree(visitor.table)
print_tree(visitor.table)
# pprint(visitor.table.symbols)
exit()
# except IdentifierRedefinition:
#     print(i)
#     print(visitor.table.symbols)
#     exit()
