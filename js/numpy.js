// JavaScript source code
var vDot = {};

class TNumpy {
    constructor() {
        this.random = new TNumpyRandom();
    }

    zeros(shape) {
        Assert(shape.length == 2, "zero-s");
        return new ArrayView(shape[0], shape[1]);
    }

    dot(A, B) {
        Assert(A instanceof ArrayView && B instanceof ArrayView && A.ncol == B.nrow && !A.columnMajor && !B.columnMajor, "d-o-t");

        var id = "" + A.nrow + "," + A.ncol + "," + B.ncol;
        if (vDot[id] == undefined) {
            vDot[id] = 0;
        }
        if (!isFloat64 && vDot[id] < 3 && A.ncol % 4 == 0) {
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
                    "A_Tex": A,
                    "B_Tex": B.T(),
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

            var diff = C1.Sub(C2).Abs().Sum() / (C1.nrow * C1.ncol);
            Assert(diff < 0.001, "dot-diff");

            console.log("dot:" + id + " tex:" + use_tex + " " + t1 + "ms  CPU:" + t2 + "ms 誤差 " + diff.toFixed(7));
            return C2;
        }

        return A.Dot(B);
    }

    argmax(x) {
        Assert(x instanceof ArrayView && x.ncol == 1, "argmax");
        var idx = x.dt.indexOf(Math.max.apply(null, x.dt));
        Assert(idx != -1, "argmax");
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

var numpy = new TNumpy();
var np = numpy;

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


