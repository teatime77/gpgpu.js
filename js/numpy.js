// JavaScript source code

var vDot = {};

function Assert(condition, message) {
    if (!condition) {
        throw new Error(message || "Assertion failed");
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
            Assert(false, "x range");
            return null;
    }

    var cnt = Math.floor((stop - start) / step);
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

function makeTextureInfo(gpgpu, texel_type, array_view) {
    var col_size;

    switch (texel_type) {
        case "float":
            return gpgpu.makeTextureInfo(texel_type, array_view.shape, array_view.dt);

        case "vec2":
            col_size = 2;
            break;

        case "vec3":
            col_size = 3;
            break;

        case "vec4":
            col_size = 4;
            break;

        default:
            Assert(false);
            break;
    }

    var shape;

    if (array_view.shape.length == 2) {

        shape = [ array_view.shape[0], array_view.shape[1] / col_size ]
    }
    else {

        shape = [ array_view.shape[0], array_view.shape[1], array_view.shape[2] / col_size ]
    }

    return gpgpu.makeTextureInfo(texel_type, shape, array_view.dt);
}

function MakeFloat32Index(n) {
    var v = new Float32Array(n);
    for (var i = 0; i < n; i++) {
        v[i] = i;
    }

    return v;
}

function make2DArray(nrow, ncol, init) {
    var v;

    if (init) {
        if (init instanceof Float32Array) {

            v = init;
        }
        else {

            v = new Float32Array(init);
        }

        Assert(v.length == nrow * ncol);
    }
    else {

        v = new Float32Array(nrow * ncol);
    }

    v.nrow = nrow;
    v.ncol = ncol;

    v.shape = [nrow, ncol];

    v.T = function () {
        var m = make2DArray(this.ncol, this.nrow);
        var i1 = 0;
        for (var r = 0; r < this.ncol; r++) {
            var i2 = r;
            for (var c = 0; c < this.nrow; c++) {
                m[i1] = this[i2];
                i1++;
                i2 += this.ncol;
            }
        }

        return m;
    }

    return v;
}

class ArrayView {
    constructor() {
        var args;

        if (arguments.length == 1 && Array.isArray(arguments[0])) {

            args = arguments[0];
        }
        else {

            // 引数のリストをArrayに変換します。
            args = Array.prototype.slice.call(arguments);
        }

        // 引数の最後
        var last_arg = args[args.length - 1];
        if (typeof last_arg != 'number') {
            // 引数の最後が数値でない場合

            if (typeof last_arg == 'ArrayView') {

                this.dt = new Float32Array(last_arg.dt);
            }
            else {

                Assert(last_arg instanceof Float32Array, "is Float32Array");
                this.dt = last_arg;
            }

            args.pop();
        }

        this.shape = args;

        this.ncol = this.shape[this.shape.length - 1];
        if (this.shape.length == 1) {

            this.nrow = 1;
        }
        else {

            this.nrow = this.shape[this.shape.length - 2];
        }

        if (!this.dt) {
            this.dt = new Float32Array(this.shape.reduce((x, y) => x * y));
        }
    }

    Map(f) {
        return new ArrayView(this.nrow, this.ncol, this.dt.map(f));
    }

    T() {
        var m = new ArrayView(this.ncol, this.nrow);
        var i1 = 0;
        for (var r = 0; r < this.ncol; r++) {
            var i2 = r;
            for (var c = 0; c < this.nrow; c++) {
                m.dt[i1] = this.dt[i2];
                i1++;
                i2 += this.ncol;
            }
        }

        return m;
    }

    At2(r, c) {
        Assert(r < this.nrow && c < this.ncol, "ArrayView-at");
        return this.dt[r * this.ncol + c];
    }

    Set2(r, c, val) {
        Assert(r < this.nrow && c < this.ncol, "ArrayView-set");

        this.dt[r * this.ncol + c] = val;
    }

    At3(d, r, c) {
        Assert(d < this.shape[this.shape.length - 3] && r < this.nrow && c < this.ncol, "ArrayView-at3");

        return this.dt[(d * this.nrow + r) * this.ncol + c];
    }

    Set3(d, r, c, val) {
        Assert(d < this.shape[this.shape.length - 3] && r < this.nrow && c < this.ncol, "ArrayView-set3");

        this.dt[(d * this.nrow + r) * this.ncol + c] = val;
    }

    Col(c) {
        var v = new Float32Array(this.nrow);
        for (var r = 0; r < this.nrow; r++) {
            v[r] = this.dt[r * this.ncol + c];
        }

        return new ArrayView(this.nrow, 1, v);
    }

    Add(m) {
        Assert(m instanceof ArrayView && m.nrow == this.nrow && m.ncol == this.ncol, "ArrayView-add");
        var v = new Float32Array(this.nrow * this.ncol);
        for (var r = 0; r < this.nrow; r++) {
            for (var c = 0; c < this.ncol; c++) {
                var k = r * this.ncol + c;
                v[k] = this.dt[k] + m.dt[k];
            }
        }

        return new ArrayView(this.nrow, this.ncol, v);
    }

    AddVec(vec) {
        Assert(vec instanceof ArrayView && vec.nrow == this.nrow && vec.ncol == 1, "ArrayView-add-V");
        var v = new Float32Array(this.nrow * this.ncol);
        for (var r = 0; r < this.nrow; r++) {
            for (var c = 0; c < this.ncol; c++) {
                var k = r * this.ncol + c;
                v[k] = this.dt[k] + vec.dt[r];
            }
        }

        return new ArrayView(this.nrow, this.ncol, v);
    }

    Reduce(f) {
        var v = new Float32Array(this.nrow);
        for (var r = 0; r < this.nrow; r++) {
            var x;
            for (var c = 0; c < this.ncol; c++) {
                var k = r * this.ncol + c;
                if (c == 0) {

                    x = this.dt[k];
                }
                else {

                    x = f(x, this.dt[k]);
                }
            }
            v[r] = x;
        }

        return new ArrayView(this.nrow, 1, v);
    }

    Sub(m) {
        Assert(m instanceof ArrayView && m.nrow == this.nrow && m.ncol == this.ncol, "ArrayView-Sub");
        var v = new Float32Array(this.nrow * this.ncol);
        for (var r = 0; r < this.nrow; r++) {
            for (var c = 0; c < this.ncol; c++) {
                var k = r * this.ncol + c;
                v[k] = this.dt[k] - m.dt[k];
            }
        }

        return new ArrayView(this.nrow, this.ncol, v);
    }

    Mul(m) {
        if (m instanceof Number) {

            return new ArrayView(this.nrow, this.ncol, this.dt.map(x => x * m));
        }
        Assert(m instanceof ArrayView && m.nrow == this.nrow && m.ncol == this.ncol, "Array-View-mul");
        var v = new Float32Array(this.nrow * this.ncol);
        for (var r = 0; r < this.nrow; r++) {
            for (var c = 0; c < this.ncol; c++) {
                var k = r * this.ncol + c;
                v[k] = this.dt[k] * m.dt[k];
            }
        }

        return new ArrayView(this.nrow, this.ncol, v);
    }

    Dot(m) {
        Assert(m instanceof ArrayView && m.nrow == this.ncol, "ArrayView-Dot");

        var v = new Float32Array(this.nrow * m.ncol);
        for (var r = 0; r < this.nrow; r++) {
            for (var c = 0; c < m.ncol; c++) {
                var sum = 0;
                for (var k = 0; k < this.ncol; k++) {
                    sum += this.dt[r * this.ncol + k] * m.dt[k * m.ncol + c];
                }
                v[r * m.ncol + c] = sum;
            }
        }
        return new ArrayView(this.nrow, m.ncol, v);
    }
}

class TNumpy {
    constructor() {
        this.random = new TNumpyRandom();
    }

    dot(A, B) {
        Assert(A instanceof ArrayView && B instanceof ArrayView && A.ncol == B.nrow, "d-o-t");

        var id = "" + A.nrow + "," + A.ncol + "," + B.ncol;
        if (vDot[id] == undefined) {
            vDot[id] = 0;
        }
        if (vDot[id] < 3 && A.ncol % 4 == 0) {
            vDot[id]++;

            var use_tex = (10 * 12 * 30 < A.nrow * A.ncol * B.ncol);
            var dot_val = new Float32Array(A.nrow * B.ncol);

            var param = {};

            param.elementCount = A.nrow * B.ncol;

            var vs_id;
            if (use_tex) {

                vs_id = "vs-Texture";
                param.args = {
                    "idx_f": MakeFloat32Index(param.elementCount),
                    "A_Tex": makeTextureInfo(WebGL2, "vec4", A),
                    "B_Tex": makeTextureInfo(WebGL2, "vec4", B.T()),
                    "B_Cols": B.ncol,
                    'dot_val': dot_val,
                };
            }
            else {
                vs_id = "vs-Uniform";

                param.args = {
                    "idx_f": MakeFloat32Index(param.elementCount),
                    "B_Cols": B.ncol,
                    "A": A,
                    "B": B.T(),
                    'dot_val': dot_val,
                };
            }

            var A_len = (A.nrow * A.ncol / 4).toString();
            var B_len = (B.nrow * B.ncol / 4).toString();
            var repeat = (A.ncol / 4).toString();
            //        console.log("A_len:[" + A_len + "] B_len:[" + B_len + "] repeat:[" + repeat + "]");

            param.vertexShader = Shaders[vs_id].replace(/_repeat_/g, repeat).replace(/_A_len_/g, A_len).replace(/_B_len_/g, B_len);
            param.id = vs_id + ":" + A.nrow + "," + A.ncol + "," + B.nrow + "," + B.ncol;

            var startTime = new Date();
            WebGL2.compute(param);
            var C1 = new ArrayView(A.nrow, B.ncol, dot_val);

            var t1 = new Date() - startTime;

            startTime = new Date();
            var C2 = A.Dot(B);
            var t2 = new Date() - startTime;

            var diff = C1.Sub(C2).dt.map(a => Math.abs(a)).reduce((x, y) => x + y) / (C1.nrow * C1.ncol);
            Assert(diff < 0.001, "dot-diff");

            console.log("dot:" + id + " tex:" + use_tex + " " + t1 + "ms  CPU:" + t2 + "ms 誤差 " + diff.toFixed(7));
            return C2;
        }

        return A.Dot(B);
    }

    argmax(x) {
        Assert(x instanceof ArrayView && x.ncol == 1, "arg max");
        var idx = x.dt.indexOf(Math.max.apply(null, x.dt));
        Assert(idx != -1, "arg max");
        return idx;
    }
}

class TNumpyRandom {
    constructor() {
        this.Flag = false;
    }

    NextDouble() {
        this.Flag = ! this.Flag;
        if (this.Flag) {
            this.C = Math.sqrt(-2 * Math.log(Math.random()));
            this.Theta = Math.random() * Math.PI * 2;

            return this.C * Math.sin(this.Theta);
        }
        else {
            return this.C * Math.cos(this.Theta);
        }
    }

    randn() {
        var m;

        switch (arguments.length) {
            case 0:
                return this.NextDouble();

            case 1:
                m = new ArrayView(1, arguments[0]);
                break;

            case 2:
                m = new ArrayView(arguments[0], arguments[1]);
                break;

            default:
                Assert(false, "");
                return null;
        }

        m.dt = m.dt.map(x => this.NextDouble());
//        m.dt = m.dt.map(x => Math.random());

        return m;
    }

    // min から max までの乱整数を返す関数
    // Math.round() を用いると、非一様分布になります!
    getRandomInt(min, max) {
        return Math.floor(Math.random() * (max - min + 1)) + min;
    }

    RandomSampling(all_count, sample_count) {
        var ret = new Array(sample_count);

        var numbers = xrange(all_count);

        for (var i = 0; i < sample_count; i++) {
            var n = this.getRandomInt(0, all_count - i - 1);

            ret[i] = numbers[n];
            numbers[n] = numbers[all_count - i - 1];
        }

        //for (var i = 0; i < sample_count; i++) {
        //    for (var j = i + 1; j < sample_count; j++) {
        //        Assert(ret[i] != ret[j], "Random-Sampling");
        //    }
        //}

        return ret;
    }

    shuffle(v) {
        var v2 = v.slice(0);
        var idx = this.RandomSampling(v.length, v.length);
        for (var i = 0; i < v.length; i++) {
            v[i] = v2[idx[i]];
        }
    }
}

var np = new TNumpy();

/*
正規乱数のテスト
var v = new Int32Array(200);
for (var i = 0; i < 10000000; i++) {
    var k = Math.floor(np.random.randn() * 25 + v.length / 2);
    if (0 <= k && k < v.length) {
        v[k]++;
    }
}
for (var k = 0; k < v.length; k++) {
    console.log("%d", v[k]);
}
*/


