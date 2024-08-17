# Factorial Test

let cnt = 5;
let fac = 1;

entry loop;

let fac = fac * cnt;
let cnt = cnt - 1;

if cnt > 1 goto loop;

ret fac;
