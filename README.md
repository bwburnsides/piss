# Python IDL Semantic Scrutinizer (PISS)

A Python frontend for parsing an IDL flavor.

## Grammar

### Keyword
```
Keyword = "const"
        | "struct"
        | "enum"
        | "typedef"
        | "module"
        | "uint"
        | "int"
```

### Integer
```
| [0-9_]+
```

### Expression
```
Expression = Identiifer | Integer
```

### PrimitiveType
```
PrimitiveType = Keyword::INT, Keyword::UINT
```

### Type
```
Type = PrimitiveType
     | Identifier
     | Type LeftBracket Expression RightBracket 
```

### Field
```
Field = Type Identifier
```

### ConstDefinition
```
ConstDefinition = Keyword::CONST Field Equals Expression SemiColon
```

### StructDefinition
```
StructDefinition = Keyword::STRUCT Identifier Identifier LeftBrace (Field Comma)* RightBrace Semicolon
```

### EnumDefinition
```
EnumDefinition = Keyword::ENUM Identifier LeftBrace (Identifier Comma)* RightBrace SemiColon
```

### TypedefDefinition
```
Keyword::TYPEDEF Type Identifier SemiColon
```

### Definition
```
Definition = ConstDefinition
           | StructDefinition
           | EnumDefinition
           | TypedefDefinition
```

### Module
```
Module = Keyword::MODULE Identifier LeftBrace Definition* RightBrace SemiColon
```