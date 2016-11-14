Number.prototype.Mul = function (x) {
    if (x instanceof Number) {
        return this * x;
    }
    else if (x instanceof Mat) {
        return x.Mul(this);
    }
    Assert(false, "Number-Mul");
    return null;
};

function Assert(b, msg) {
    if (!b) {
        console.log(msg + " " + this.LineText);
    }
}

function len(x) {
    return x.length;
}

function sum(x) {
    Assert(x instanceof Array, "sum");
    return x.reduce((x, y) => x + y);
}

function SetAt(v, i, x) {
    Assert(v instanceof Array, "Set-At");
    if (0 <= i) {
        v[i] = x;
    }
    else {
        v[v.length + i] = x;
    }
}

function xrange() {
    var start, stop, step;

    switch (arguments.length) {
        case 1:
            start = 0;
            stop = arguments[0];
            step = 1;
            break;

        case 2:
            start = arguments[0];
            stop = arguments[1];
            step = 1;
            break;

        case 3:
            start = arguments[0];
            stop = arguments[1];
            step = arguments[2];
            break;

        default:
            Assert(false, "Slice");
            return null;
    }

    var cnt = Math.floor( (stop - start) / step );
    Assert(cnt * step == stop - start, "x-range");
/*
    var list = new Int32Array(cnt);
    var k = 0;
    for (i = start; i < stop; i += step) {
        list[k] = i;
        k++;
    }

    var list = new Array();
    for (i = start; i < stop; i += step) {
        list.push(i);
    }
*/

    var list = new Array(cnt);
    var k = 0;
    for (i = start; i < stop; i += step) {
        list[k] = i;
        k++;
    }

    return list;
}

function zip() {
    var list = new Array();

    for (var i = 0; ; i++) {
        var tpl = new Array();
        for (var k = 0; k < arguments.length; k++) {
            var arg = arguments[k];
            if (arg.length <= i) {
                return list;
            }
            tpl.push(arg[i]);
        }
        list.push(tpl);
    }
}

function zip2(u, v, f) {
    Assert(u instanceof Array && v instanceof Array && u.length == v.length, "zip2");

    var ret = new Array();
    for (var i = 0; i < u.length; i++) {
        ret.push(f(u[i], v[i]));
    }

    return ret;
}

function copyArray(target, target_start, source, source_start, source_end) {
    Assert(target instanceof Array && source instanceof Array, "copy-Array");
    if (!source_start) {
        source_start = 0;
    }
    if (!source_end) {
        source_end = source.length;
    }
    var i = target_start;

    for (var j = source_start; j < source_end; j++) {
        target[i] = source[j];
        i++;
    }
}

function Slice(v) {
    Assert(arguments.length == 2, "Slice");
    var t = arguments[1];

    var st, ed;
    switch (t.length) {
        case 1:
            return v.slice(t[0]);

        case 2:
            st = (t[0] == null ? 0 : t[0]);
            ed = (0 <= t[1] ? t[1] : v.length + t[1]);
            return v.slice(st, ed);

        case 3:
            Assert(false, "Slice 未実装");
            return null;

        default:
            Assert(false, "Slice");
            return null;
    }
}
