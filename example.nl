let x = 1;
let y = 2;
let x = x + y;        # let <ident> = <expr> ;

if x > 2 goto fINAl;  # if <expr> goto <ident> ;

let x = 1;

entry fINAL;          # entry <ident> ;

ret x;                # ret <expr> ;

let res = add (2, 3);