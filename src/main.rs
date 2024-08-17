use std::{cell::RefCell, collections::HashMap, fs, rc::{Rc, Weak}, env};

// Token


#[derive(Debug, Clone)]
enum Keyword {
    Let, Def, Func, Begin, End, Ret, Ref, Entry, Goto, If, Else,
}

#[derive(Debug, Clone)]
enum Symbol {
    Semicolon,    // ;
    Colon,        // :
    Period,       // ,
    Comma,        // .
    Assign,       // =
    Equal,        // ==
    Add,          // +
    Dash,         // -
    Asterisk,     // *
    Slash,        // /
    LParen,       // (
    RParen,       // )
    Lesser,       // <
    Greater,      // >
    LesserEqual,  // <=
    GreaterEqual, // >=
    LArrow,       // <-
    RArrow,       // ->
    SQuote,       // '
}



#[derive(Debug, Clone)]
enum Token {
    Keyword(Keyword),
    Symbol(Symbol),
    Identifier(String),
    Value(String),
    String(String),
}


// Char Type

enum CharType {
    Name,
    Number,
    Dot,
    Symbol,
    Quote,
    Hashtag,
    Void,
    Invalid,
}

trait CheckCharType {
    fn check(self) -> CharType;
}

impl CheckCharType for u8 {
    fn check(self) -> CharType {
        match self as u8 {
            65 ..= 90 | 97 ..= 122 | 95 
                => CharType::Name,   // A - Z, a - z, _
            48 ..= 57
                => CharType::Number, // 0 - 9, .
            46  => CharType::Dot,
            33 | 36 ..= 45 | 47 | 58 ..= 64 | 91 ..= 94 | 96 | 123 ..= 126
                => CharType::Symbol, // Double Quote, Hashtag not included
            34
                => CharType::Quote,
            35
                => CharType::Hashtag,
            10 | 32
                => CharType::Void,   // Space, Endl
            _   => CharType::Invalid,
        }
    }
}


// Lexer

struct Lexer {
    keyword_map: HashMap<&'static str, Keyword>,
    symbol_map:  HashMap<&'static str, Symbol>,
}

impl Lexer {
    fn new() -> Self {
        Lexer { 
            keyword_map: HashMap::from( [
                ("let",   Keyword::Let),
                ("def",   Keyword::Def),
                ("func",  Keyword::Func),
                ("begin", Keyword::Begin),
                ("end",   Keyword::End),
                ("ret",   Keyword::Ret),
                ("ref",   Keyword::Ref),
                ("entry", Keyword::Entry),
                ("goto",  Keyword::Goto),
                ("if",    Keyword::If),
                ("else",  Keyword::Else),
            ] ),
            symbol_map: HashMap::from( [
                (";",  Symbol::Semicolon),
                (":",  Symbol::Colon),
                (",",  Symbol::Comma),
                (".",  Symbol::Period),
                ("=",  Symbol::Assign),
                ("==", Symbol::Equal),
                ("+",  Symbol::Add),
                ("-",  Symbol::Dash),
                ("*",  Symbol::Asterisk),
                ("/",  Symbol::Slash),
                ("(",  Symbol::LParen),
                (")",  Symbol::RParen),
                ("<",  Symbol::Lesser),
                (">",  Symbol::Greater),
                ("<=", Symbol::LesserEqual),
                (">=", Symbol::GreaterEqual),
                ("->", Symbol::RArrow),
                ("<-", Symbol::LArrow),
                ("'",  Symbol::SQuote),
            ] )
        }
    }

    // String should be ascii, ended with endl
    fn lex(&self, text: String) -> Vec<Token> {
        let bytes = text.as_bytes();
        let mut tokens: Vec<Token> = Vec::new();

        let mut now: usize = 0;
        let mut pre: usize;
        

        while now < bytes.len() {
            match bytes[now].check() {

                CharType::Void => { 
                    now += 1; 
                },

                CharType::Hashtag => {
                    while bytes[now] != '\n' as u8 { now += 1; }
                    now += 1;
                },

                CharType::Name => {
                    pre = now;
                    while let CharType::Name | CharType::Number = bytes[now].check() { now += 1; }
                    let name = &text[pre..now];
                    if let Some(keyword) = self.keyword_map.get(name) {
                        tokens.push(Token::Keyword(keyword.clone()));
                    } else {
                        tokens.push(Token::Identifier(String::from(name)));
                    }
                },

                CharType::Number | CharType::Dot => {
                    pre = now;
                    while let CharType::Number | CharType::Dot = bytes[now].check() { now += 1; }
                    tokens.push(Token::Value(String::from(&text[pre..now])));
                },

                CharType::Symbol   => {
                    pre = now;
                    while let CharType::Symbol = bytes[now].check() { now += 1; }

                    while pre < now {
                        let mut pos: usize = now;
                        while let None = self.symbol_map.get(&text[pre..pos]) { pos -= 1; }
                        tokens.push(Token::Symbol(self.symbol_map[&text[pre..pos]].clone()));
                        pre = pos;
                    }
                },

                CharType::Quote    => {
                    pre = now + 1;
                    now += 1;
                    loop {
                        if let CharType::Quote = bytes[now].check() { break; }
                        now += 1;
                    }
                    tokens.push(Token::String(String::from(&text[pre..now])));
                    now += 1;
                },

                CharType::Invalid  => panic!("Lexer Error: Invalid Symbol!"),

            }
        }

        return tokens;
    }
}


// Object

#[derive(Clone, Debug)]
enum Object {
    Void,
    Bool(bool),
    Int(i32),
    Fp(f32),
    Str(String),
}

type Registry = HashMap<String, Rc<Object>>;


// Statement

trait Statement {
    fn execute(&self, reg: &mut Registry) -> Option<Rc<RefCell<dyn Statement>>>;
    fn set_next(&mut self, target: Option<Rc<RefCell<dyn Statement>>>);
}

struct LetStatement {
    identifier: String,
    expression: Box<dyn Expression>,
    next: Option<Rc<RefCell<dyn Statement>>>,
}

impl Statement for LetStatement {
    fn execute(&self, reg: &mut Registry) -> Option<Rc<RefCell<dyn Statement>>> {
        reg.insert(self.identifier.clone(), Rc::new(self.expression.evaluate(reg)));
        return self.next.clone();
    }

    fn set_next(&mut self, target: Option<Rc<RefCell<dyn Statement>>>) {
        self.next = target;
    }
}


struct GotoStatement {
    condition: Box<dyn Expression>,
    jump: Option<Weak<RefCell<EntryStatement>>>,
    next: Option<Rc<RefCell<dyn Statement>>>,
}

impl Statement for GotoStatement {
    fn execute(&self, reg: &mut Registry) -> Option<Rc<RefCell<dyn Statement>>> {
        match self.condition.evaluate(reg) {
            Object::Bool(true)  => Some(self.jump.clone().unwrap().upgrade().unwrap()),
            Object::Bool(false) => self.next.clone(),
            _ => panic!("GotoStatement: Condition not Bool Type!"),
        }
    }

    fn set_next(&mut self, target: Option<Rc<RefCell<dyn Statement>>>) {
        self.next = target;
    }
}

struct EntryStatement {
    next: Option<Rc<RefCell<dyn Statement>>>,
}

impl Statement for EntryStatement {
    fn execute(&self, _reg: &mut Registry) -> Option<Rc<RefCell<dyn Statement>>> {
        return self.next.clone();
    }

    fn set_next(&mut self, target: Option<Rc<RefCell<dyn Statement>>>) {
        self.next = target;
    }
}


// Expression

trait Expression {
    fn evaluate(&self, reg: &Registry) -> Object;
}

struct IdentifierExpression {
    identifier: String,
}

impl Expression for IdentifierExpression {
    fn evaluate(&self, reg: &Registry) -> Object {
        return (**reg.get(&self.identifier).expect("IdentifierExpression: Identifier not Registered!")).clone();
    }
}

struct LiteralExpression {
    literal: Object,
}

impl Expression for LiteralExpression {
    fn evaluate(&self, _reg: &Registry) -> Object {
        return self.literal.clone();
    }
}

struct NegateOperation {
    expr: Box<dyn Expression>,
}

impl Expression for NegateOperation {
    fn evaluate(&self, reg: &Registry) -> Object {
        match self.expr.evaluate(reg) {
            Object::Int(i) => Object::Int(-i),
            Object::Fp(f) => Object::Fp(-f),
            _ => panic!("NegateOperation: Invalid types!"),
        }
    }
}

struct EqualOperation {
    left:  Box<dyn Expression>,
    right: Box<dyn Expression>,
}

impl Expression for EqualOperation {
    fn evaluate(&self, reg: &Registry) -> Object {
        let left_obj  = (*self.left).evaluate(reg);
        let right_obj = (*self.right).evaluate(reg);

        match (left_obj, right_obj) {
            (Object::Int(l), Object::Int(r)) 
                => Object::Bool(l == r),
            (Object::Fp(l), Object::Fp(r))
                => Object::Bool(l == r),
            _   => panic!("GreaterEqualOperation: Invalid types!")
        }
    }
}

struct GreaterEqualOperation {
    left:  Box<dyn Expression>,
    right: Box<dyn Expression>,
}

impl Expression for GreaterEqualOperation {
    fn evaluate(&self, reg: &Registry) -> Object {
        let left_obj  = (*self.left).evaluate(reg);
        let right_obj = (*self.right).evaluate(reg);

        match (left_obj, right_obj) {
            (Object::Int(l), Object::Int(r)) 
                => Object::Bool(l >= r),
            (Object::Fp(l), Object::Fp(r))
                => Object::Bool(l >= r),
            _   => panic!("GreaterEqualOperation: Invalid types!")
        }
    }
}

struct LesserOperation {
    left:  Box<dyn Expression>,
    right: Box<dyn Expression>,
}

impl Expression for LesserOperation {
    fn evaluate(&self, reg: &Registry) -> Object {
        let left_obj  = (*self.left).evaluate(reg);
        let right_obj = (*self.right).evaluate(reg);

        match (left_obj, right_obj) {
            (Object::Int(l), Object::Int(r)) 
                => Object::Bool(l < r),
            (Object::Fp(l), Object::Fp(r))
                => Object::Bool(l < r),
            _   => panic!("LesserOperation: Invalid types!")
        }
    }
}

struct AddOperation {
    left:  Box<dyn Expression>,
    right: Box<dyn Expression>,
}

impl Expression for AddOperation {
    fn evaluate(&self, reg: &Registry) -> Object {
        let left_obj  = (*self.left).evaluate(reg);
        let right_obj = (*self.right).evaluate(reg);

        match (left_obj, right_obj) {
            (Object::Int(l), Object::Int(r)) 
                => Object::Int(l + r),
            (Object::Fp(l), Object::Fp(r))
                => Object::Fp(l + r),
            _   => panic!("AddOperation: Invalid types!")
        }
    }
}

struct MulOperation {
    left:  Box<dyn Expression>,
    right: Box<dyn Expression>,
}

impl Expression for MulOperation {
    fn evaluate(&self, reg: &Registry) -> Object {
        let left_obj  = (*self.left).evaluate(reg);
        let right_obj = (*self.right).evaluate(reg);

        match (left_obj, right_obj) {
            (Object::Int(l), Object::Int(r)) 
                => Object::Int(l * r),
            (Object::Fp(l), Object::Fp(r))
                => Object::Fp(l * r),
            _   => panic!("AddOperation: Invalid types!")
        }
    }
}

struct DivOperation {
    left:  Box<dyn Expression>,
    right: Box<dyn Expression>,
}

impl Expression for DivOperation {
    fn evaluate(&self, reg: &Registry) -> Object {
        let left_obj  = (*self.left).evaluate(reg);
        let right_obj = (*self.right).evaluate(reg);

        match (left_obj, right_obj) {
            (Object::Int(l), Object::Int(r)) 
                => Object::Int(l / r),
            (Object::Fp(l), Object::Fp(r))
                => Object::Fp(l / r),
            _   => panic!("AddOperation: Invalid types!")
        }
    }
}


// Program

struct Program {
    reg: Registry,
    root: Option<Rc<RefCell<dyn Statement>>>,
}

impl Program {
    fn new() -> Self {
        Program {
            reg: HashMap::new(),
            root: Some(Rc::new(RefCell::new(EntryStatement{next: None}))),
        }
    }


    fn execute(&mut self, arg: Object) -> Object {
        self.reg.insert(String::from("arg"), Rc::new(arg));
        let mut curr = self.root.clone();
        while let Some(ptr) = curr {
            curr = (*ptr).borrow().execute(&mut self.reg);
        }
        return (**self.reg.get("ret").unwrap()).clone();
    }
}


// Expression Parser



fn parse_value(value: String) -> Object {
    match value.parse::<i32>() {
        Ok(i) => Object::Int(i),
        Err(_) => {
            if let Ok(f) = value.parse::<f32>() {
                Object::Fp(f)
            } else {
                panic!("ValueParser: Failure!")
            }
        }
    }
}

fn get_prefix_precedence(op: Symbol) -> i8 {
    match op {
        Symbol::Add | Symbol::Dash => 17,
        _ => panic!("PrefixOperationPrecedence: Invalid Symbol!"),
    }
}

fn get_infix_precedence(op: Symbol) -> (i8, i8) {
    match op {
        Symbol::Equal        => (7,  8),
        Symbol::Greater      => (9,  10),
        Symbol::Lesser       => (9,  10),
        Symbol::GreaterEqual => (9,  10),
        Symbol::LesserEqual  => (9,  10),
        Symbol::Add          => (11, 12),
        Symbol::Dash         => (11, 12),
        Symbol::Asterisk     => (13, 14),
        Symbol::Slash        => (13, 14),
        // Symbol::Precentage => (13, 14),
        // Symbol::Power => (15, 16),

        Symbol::RParen       => (-1, 0),
        _ => panic!("InfixOperationPrecedence: Invalid Symbol!"),
    }
}

struct ExpressionParser<'a> {
    tokens: &'a [Token],
    index: usize,
}

impl<'a> ExpressionParser<'a> {
    fn new(_tokens: &'a [Token]) -> Self {
        ExpressionParser { index: 0, tokens: _tokens }
    }

    fn parse_expression(&mut self) -> Box<dyn Expression> {
        self.index = 0;
        self.parse_expression_recursive(0)
    }

    fn parse_expression_recursive(&mut self, min_precedence: i8) -> Box<dyn Expression> {
        
        let mut root: Box<dyn Expression> = Box::new(LiteralExpression{ literal: Object::Void });
        let mut fresh: bool =  true;

        // Assume no infix after fresh! If that goes wrong then everything fuck up.

        while self.index < self.tokens.len() { // Not sure < or <= yet

            let curr = self.tokens.get(self.index).unwrap().clone();
            self.index += 1;

            match curr {

                Token::Symbol(symbol) => {
                    if !fresh && get_infix_precedence(symbol.clone()).0 < min_precedence {
                        break;
                    }

                    match symbol {
                        Symbol::LParen => {
                            if fresh {
                                root = self.parse_expression_recursive(0);
                            } else {
                                panic!{"ExpressionParser: Left Parenthesis is not an infix operator!"}
                            }
                        }

                        Symbol::Add => {
                            if fresh { // Prefix
                                root = self.parse_expression_recursive(get_prefix_precedence(Symbol::Add));
                            } else {
                                root = Box::new(AddOperation{ left: root, right: self.parse_expression_recursive(get_infix_precedence(Symbol::Add).1) });
                            }
                        },

                        Symbol::Dash => {
                            if fresh { // Prefix
                                root = Box::new(NegateOperation{ expr: self.parse_expression_recursive(get_prefix_precedence(Symbol::Dash)) });
                            } else {
                                root = Box::new(AddOperation { left: root, right: Box::new(NegateOperation{ expr: self.parse_expression_recursive(get_infix_precedence(Symbol::Dash).1)}) });
                            }
                        },

                        Symbol::Asterisk => {
                            root = Box::new(MulOperation{ left: root, right: self.parse_expression_recursive(get_infix_precedence(Symbol::Asterisk).1) });
                        },

                        Symbol::Slash => {
                            root = Box::new(DivOperation{ left: root, right: self.parse_expression_recursive(get_infix_precedence(Symbol::Slash).1) });
                        },

                        Symbol::Equal => {
                            root = Box::new(EqualOperation{ right: root, left: self.parse_expression_recursive(get_infix_precedence(Symbol::Equal).1) })
                        },

                        Symbol::Greater => {
                            root = Box::new(LesserOperation{ right: root, left: self.parse_expression_recursive(get_infix_precedence(Symbol::Greater).1) })
                        },

                        Symbol::Lesser => {
                            root = Box::new(LesserOperation{ left: root, right: self.parse_expression_recursive(get_infix_precedence(Symbol::Lesser).1) })
                        },

                        Symbol::GreaterEqual => {
                            root = Box::new(GreaterEqualOperation{ left: root, right: self.parse_expression_recursive(get_infix_precedence(Symbol::GreaterEqual).1) })
                        },

                        Symbol::LesserEqual => {
                            root = Box::new(GreaterEqualOperation{ right: root, left: self.parse_expression_recursive(get_infix_precedence(Symbol::LesserEqual).1) })
                        },

                        _ => panic!("ExpressionParser: Invalid Symbol!"),
                    }
                },

                Token::Identifier(identifier) => {
                    root = Box::new(IdentifierExpression{identifier: identifier});
                },

                Token::String(string) => {
                    todo!();
                },

                Token::Value(value) => {
                    root = Box::new(LiteralExpression{literal: parse_value(value)});
                },

                _ => panic!("ExpressionParser: Invalid Token!"),
            }

            fresh = false;

        }
        
        return root;

    }
}


// Parser

struct Parser {
    goto_list:  Vec<(String, Rc<RefCell<GotoStatement>>)>,
    entry_list: HashMap<String, Rc<RefCell<EntryStatement>>>,
}

impl Parser {
    fn new() -> Self {
        Parser {
            goto_list:   Vec::new(),
            entry_list:  HashMap::new(),
        }
    }

    fn parse(&mut self, tokens: Vec<Token>) -> Program {
        let prgm: Program = Program::new();
        let mut curr = prgm.root.clone().unwrap();

        let mut now: usize = 0;

        while now < tokens.len() {
            match tokens[now] {
                Token::Keyword(Keyword::Let) => {
                    let ident: String;

                    now += 1;
                    if let Token::Identifier(_ident) = &tokens[now] { ident = _ident.clone(); }
                    else { panic!("Parser: Second Token of Let Statement is not an identifier!"); }

                    now += 1;
                    if let Token::Symbol(Symbol::Assign) = tokens[now] { } 
                    else { panic!("Parser: Third Token of Let Statement is not assign symbol!"); }

                    let pre = now + 1;
                    loop {
                        now += 1;
                        if let Token::Symbol(Symbol::Semicolon) = tokens[now] { break; }
                    }

                    let mut expr_parser = ExpressionParser::new(&tokens[pre..now]);
                    let expr = expr_parser.parse_expression();
                    now += 1;

                    let new = Rc::new(RefCell::new(LetStatement { identifier: ident, expression: expr, next: None }));
                    (*curr).borrow_mut().set_next(Some(new.clone()));

                    curr = new;
                },

                Token::Keyword(Keyword::Ret) => {
                    let pre = now + 1;
                    loop {
                        now += 1;
                        if let Token::Symbol(Symbol::Semicolon) = tokens[now] { break; }
                    }

                    let mut expr_parser = ExpressionParser::new(&tokens[pre..now]);
                    let expr = expr_parser.parse_expression();
                    now += 1;

                    let new = Rc::new(RefCell::new(LetStatement { identifier: String::from("ret"), expression: expr, next: None }));
                    (*curr).borrow_mut().set_next(Some(new.clone()));

                    curr = new;
                },

                Token::Keyword(Keyword::Goto) => {
                    let entry: String;

                    now += 1;
                    if let Token::Identifier(_entry) = &tokens[now] { entry = _entry.clone(); }
                    else { panic!("Parser: Second Token of Goto Statement is not an identifier!") }

                    now += 1;
                    if let Token::Symbol(Symbol::Semicolon) = &tokens[now] { }
                    else { panic!("Parser: Third Token of Goto Statement is not Semicolon!") }

                    let new = Rc::new(RefCell::new(GotoStatement { condition: Box::new(LiteralExpression { literal: Object::Bool(true) }), jump: None, next: None}));
                    self.goto_list.push((entry, new.clone()));

                    (*curr).borrow_mut().set_next(Some(new.clone()));

                    curr = new;

                    now += 1;
                },

                Token::Keyword(Keyword::If) => {
                    let pre = now + 1;
                    loop {
                        now += 1;
                        if let Token::Keyword(Keyword::Goto) = tokens[now] { break; }
                    }

                    let mut expr_parser = ExpressionParser::new(&tokens[pre..now]);
                    let expr = expr_parser.parse_expression();

                    now += 1;
                    let entry: String;
                    if let Token::Identifier(_entry) = &tokens[now] { entry = _entry.clone(); }
                    else { panic!("Parser: In IfGoto Statement, the Token after Goto is not an identifier!") }

                    now += 1;
                    if let Token::Symbol(Symbol::Semicolon) = &tokens[now] { }
                    else { panic!("Parser: IfGoto Statement not ended correctly.") }

                    let new = Rc::new(RefCell::new(GotoStatement { condition: expr, jump: None, next: None}));
                    self.goto_list.push((entry, new.clone()));

                    (*curr).borrow_mut().set_next(Some(new.clone()));

                    curr = new;

                    now += 1;
                },

                Token::Keyword(Keyword::Entry) => {
                    let entry: String;

                    now += 1;
                    if let Token::Identifier(_entry) = &tokens[now] { entry = _entry.clone(); }
                    else { panic!("Parser: Second Token of Entry Statement is not an identifier!") }

                    now += 1;
                    if let Token::Symbol(Symbol::Semicolon) = &tokens[now] { }
                    else { panic!("Parser: Third Token of Entry Statement is not Semicolon!") }

                    let new = Rc::new(RefCell::new(EntryStatement { next: None }));
                    self.entry_list.insert(entry, new.clone());

                    (*curr).borrow_mut().set_next(Some(new.clone()));

                    curr = new;

                    now += 1;
                },

                _ => panic!("Parser: Unknown Statement!"),

            }
        }

        self.solve_goto();
        
        return prgm;
    }

    fn solve_goto(&mut self) {

        while let Some((name, stmt)) = self.goto_list.pop() {
            (*stmt).borrow_mut().jump = Some(Rc::downgrade(self.entry_list.get(&name).expect("GotoSolver: Entry name not found!")).clone());
        }

        self.entry_list.clear();
    }

}





fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        println!("Usage: nooblang [path to file]");
        return;
    }

    let mut src = fs::read_to_string(args[1].clone()).unwrap();
    src.push('\n');

    let lexer: Lexer = Lexer::new();
    let tokens = lexer.lex(src);
    println!("-----Tokens-----\n{:?}\n---------------", tokens);

    let mut parser = Parser::new();
    let mut prgm = parser.parse(tokens);

    let res = prgm.execute(Object::Void);

    println!("Result: {:?}", res);
}
