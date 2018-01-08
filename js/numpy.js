// JavaScript source code

var vsTextureShader = `
uniform int B_Cols;

uniform sampler2D A_Tex;
uniform sampler2D B_Tex;

in float idx_f;

out float dot_val;

void main() {
    uint idx = uint(idx_f);
    int i   = int(idx / uint(B_Cols));
    int j   = int(idx % uint(B_Cols));

    int k;
    float sum = 0.0;
    for(k = 0; k < _repeat_; k++) {
        vec4  A_txl, B_txl;

        A_txl = texelFetch(A_Tex, ivec2(k, i), 0);
        B_txl = texelFetch(B_Tex, ivec2(k, j), 0);
        sum   += dot(A_txl, B_txl);
    }

    dot_val = sum;
}`;

var vsUniformShader =
`
uniform int B_Cols;

uniform vec4 A[_A_len_];
uniform vec4 B[_B_len_];

in float idx_f;

out float dot_val;

void main() {
    uint idx = uint(idx_f);
    int i   = int(idx / uint(B_Cols));
    int j   = int(idx % uint(B_Cols));

    int k;
    float sum = 0.0;
    for(k = 0; k < _repeat_; k++) {
        sum += dot(A[_repeat_*i +k], B[_repeat_*j +k]);
    }
    dot_val = sum;
}`;

var vDot = {};


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

class ArrayViewOLD {

    T() {
        Assert(this.shape.length == 2, "array-view-t")
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

    Dot2(gpgpu, m) {
        var vertex_shader =
           `in float zero;

       // 2次元配列のテクスチャ
        uniform sampler2D A;
        uniform sampler2D B;

        // 出力変数C
        out float C;

        void main() {
            // テクスチャBの行数と列数を取得します。
            // B_sz.yが行数、B_sz.xが列数です。
            ivec2 B_sz = textureSize(B, 0);

            // 出力する行列Cの行(row)と列(col)を計算します。
            // gl_VertexIDは入力変数の何番目の要素かを示すシステム変数です。
            int row = gl_VertexID / B_sz.x;
            int col = gl_VertexID % B_sz.x;

            // Cのrow行col列の値は、Aのrow行のベクトルとBのcol列のベクトルの内積です。

            // 以下のループでベクトルの内積を計算します。
            float sum = 0.0f;
            for(int i = 0; i < B_sz.y; i++) {

                // Aのrow行i列の値を取得します。
                vec4 a = texelFetch(A, ivec2(i, row), 0);

                // Bのi行col列の値を取得します。
                vec4 b = texelFetch(B, ivec2(col, i), 0);

                // a.rとb.rに取得した値が入っています。
                sum += a.r * b.r;
            }

            // 入力変数zeroの値は必要ないですが、使用しない変数はコンパイラが除去してしまいエラーになるので形の上だけ使用します。
            // zeroの値は0なので計算結果には影響しません。
            C = sum + zero;
        }`;

        Assert(this.ncol == m.nrow, "dot2");
        var C = new ArrayView(this.nrow, m.ncol);
        var param = {
            id: "dot," + this.nrow + "," + this.ncol + "," + m.ncol,
            vertexShader: vertex_shader,
            args: {
                "zero": new Float32Array(this.nrow * m.ncol),
                "A": gpgpu.makeTextureInfo("float", [this.nrow, this.ncol], this.dt),
                "B": gpgpu.makeTextureInfo("float", [   m.nrow,    m.ncol], m.dt),
                "C": C.dt,
            }
        };

        gpgpu.compute(param);

        return C;
    }
}

class TNumpy {
    constructor() {
        this.random = new RandomHelper();
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
            var vertex_shader;
            if (use_tex) {

                vs_id = "vs-Texture";
                vertex_shader = vsTextureShader;
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
                vertex_shader = vsUniformShader;
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

            param.vertexShader = vertex_shader.replace(/_repeat_/g, repeat).replace(/_A_len_/g, A_len).replace(/_B_len_/g, B_len);
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
}
