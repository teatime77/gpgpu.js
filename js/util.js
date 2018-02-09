// JavaScript source code

function Assert(condition, message) {
    if (!condition) {
        throw new Error(message || "Assertion failed");
    }
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

    /*
        指定した行を返す。
    */
    Row(r) {
        var v = new ArrayView(this.ncol);
        for (var c = 0; c < this.ncol; c++) {
            v.dt[c] = this.dt[r * this.ncol + c];
        }

        return v;
    }

    Reduce(f) {
        var v = new ArrayView(this.ncol);

        for (var c = 0; c < this.ncol; c++)  {
            var x;
            for (var r = 0; r < this.nrow; r++) {
                var k = r * this.ncol + c;
                if (r == 0) {

                    x = this.dt[k];
                }
                else {

                    x = f(x, this.dt[k]);
                }
            }
            v.dt[c] = x;
        }

        return v;
    }

    diff(m) {
        Assert(m.length == this.dt.length);
        var x = 0;
        for (var i = 0; i < this.dt.length; i++) {
            x = Math.max(x, Math.abs(m[i] - this.dt[i]));
        }

        return x;
    }
}

function argmax(x) {
    Assert(x instanceof ArrayView, "arg max");
    var idx = x.dt.indexOf(Math.max.apply(null, x.dt));
    Assert(idx != -1, "arg max");
    return idx;
}


class RandomHelper {
    constructor() {
        this.Flag = false;
    }

    NextDouble() {
        this.Flag = ! this.Flag;
        if (this.Flag) {
            this.C = Math.sqrt(-2 * Math.log(Math_random()));
            this.Theta = Math_random() * Math.PI * 2;

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
//        m.dt = m.dt.map(x => Math_random());

        return m;
    }

    // min から max までの乱整数を返す関数
    // Math.round() を用いると、非一様分布になります!
    getRandomInt(min, max) {
        return Math.floor(Math_random() * (max - min + 1)) + min;
    }

    RandomSampling(all_count, sample_count) {
        if (!sample_count) {
            // サンプリングする数が指定されてない場合

            // すべてサンプリングする。
            sample_count = all_count;
        }
        var ret = new Array(sample_count);

        var numbers = new Int32Array(all_count);
        for (var i = 0; i < all_count; i++) {
            numbers[i] = i;
        }

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
}
