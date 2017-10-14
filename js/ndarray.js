class Mat {
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

            if (typeof last_arg == 'Mat') {

                this.dt = new Float32Array(last_arg.dt);
            }
            else {

                Assert(last_arg instanceof Float32Array, "is Float32Array");
                this.dt = last_arg;
            }
            args.pop();
        }

        this.shape = args;

        switch (this.shape.length) {
            case 1:
                this.Times = 1;
                this.Depth = 1;
                this.Rows = 1;
                this.Cols = this.shape[0];
                break;

            case 2:
                this.Times = 1;
                this.Depth = 1;
                this.Rows = this.shape[0];
                this.Cols = this.shape[1];
                break;

            case 3:
                this.Times = 1;
                this.Depth = this.shape[0];
                this.Rows = this.shape[1];
                this.Cols = this.shape[2];
                break;

            case 4:
                this.Times = this.shape[0];
                this.Depth = this.shape[1];
                this.Rows = this.shape[2];
                this.Cols = this.shape[3];
                break;

            default:
                Assert(false, "new mat:" + String(this.shape.length) + ":" + typeof arguments[0] + ":" + arguments[0] + ":" + String(arguments.length))
                break;
        }

        this.nElement = this.shape.reduce((x, y) => x * y);

        if (this.dt) {

            Assert(this.dt.length == this.nElement);
        }
        else {
            this.dt = new Float32Array(this.nElement);
        }
    }

    map(f) {
        return new Mat(this.Rows, this.Cols, this.dt.map(f));
    }

    T(m) {
        if (m == undefined) {
            m = new Mat(this.Cols, this.Rows);
        }
        for (var r = 0; r < this.Cols; r++) {
            for (var c = 0; c < this.Rows; c++) {
                m.dt[r * this.Rows + c] = this.dt[c * this.Cols + r];
            }
        }

        return m;
    }

    transpose() {
        return this.T();
    }

    At(r, c) {
        Assert(r < this.Rows && c < this.Cols && this.Depth == 1, "Mat-at");
        return this.dt[r * this.Cols + c];
    }

    Set(r, c, val) {
        Assert(r < this.Rows && c < this.Cols && this.Depth == 1, "Mat-set");

        this.dt[r * this.Cols + c] = val;
    }

    Set3(d, r, c, val) {
        Assert(d < this.Depth && r < this.Rows && c < this.Cols, "Mat-set3");

        this.dt[(d * this.Rows + r) * this.Cols + c] = val;
    }

    At3(d, r, c) {
        Assert(d < this.Depth && r < this.Rows && c < this.Cols, "Mat-at3");

        return this.dt[(d * this.Rows + r) * this.Cols + c];
    }

    Col(c) {
        var v = newFloatArray(this.Rows);
        for (var r = 0; r < this.Rows; r++) {
            v[r] = this.dt[r * this.Cols + c];
        }

        return new Mat(this.Rows, 1, v);
    }

    CopyRows(m, r_src, r_dst, r_cnt) {
        Assert(m instanceof Mat && m.Cols == this.Cols && r_src + r_cnt <= this.Rows && r_dst + r_cnt <= m.Rows, "copy-rows");

        for (var r = 0; r < r_cnt; r++) {
            for (var c = 0; c < this.Cols; c++) {
                m.dt[(r_dst + r) * m.Cols + c] = this.dt[(r_src + r) * this.Cols + c];
            }
        }
    }

    CopyCols(m, c_src, c_dst, c_cnt) {
        Assert(m instanceof Mat && m.Rows == this.Rows && c_src + c_cnt <= this.Cols && c_dst + c_cnt <= m.Cols, "copy-cols");

        for (var r = 0; r < this.Rows; r++) {
            for (var c = 0; c < c_cnt; c++) {
                m.dt[ r * m.Cols + c_dst + c ] = this.dt[ r * this.Cols + c_src + c ];
            }
        }
    }

    Add(m) {
        Assert(m instanceof Mat && m.Rows == this.Rows && m.Cols == this.Cols, "Mat-add");
        var v = newFloatArray(this.Rows * this.Cols);
        for (var r = 0; r < this.Rows; r++) {
            for (var c = 0; c < this.Cols; c++) {
                var k = r * this.Cols + c;
                v[k] = this.dt[k] + m.dt[k];
            }
        }

        return new Mat(this.Rows, this.Cols, v);
    }

    AddV(m) {
        Assert(m instanceof Mat && m.Rows == this.Rows && m.Cols == 1, "Mat-add-V");
        var v = newFloatArray(this.Rows * this.Cols);
        for (var r = 0; r < this.Rows; r++) {
            for (var c = 0; c < this.Cols; c++) {
                var k = r * this.Cols + c;
                v[k] = this.dt[k] + m.dt[r];
            }
        }

        return new Mat(this.Rows, this.Cols, v);
    }

    SubV(m) {
        Assert(m instanceof Mat && m.Rows == this.Rows && m.Cols == 1, "Mat-sub-V");
        var v = newFloatArray(this.Rows * this.Cols);
        for (var r = 0; r < this.Rows; r++) {
            for (var c = 0; c < this.Cols; c++) {
                var k = r * this.Cols + c;
                v[k] = this.dt[k] - m.dt[r];
            }
        }

        return new Mat(this.Rows, this.Cols, v);
    }

    reduce(f) {
        var v = newFloatArray(this.Rows);
        for (var r = 0; r < this.Rows; r++) {
            var x;
            for (var c = 0; c < this.Cols; c++) {
                var k = r * this.Cols + c;
                if (c == 0) {

                    x = this.dt[k];
                }
                else {

                    x = f(x, this.dt[k]);
                }
            }
            v[r] = x;
        }

        return new Mat(this.Rows, 1, v);
    }

    Sub(m) {
        Assert(m instanceof Mat && m.Rows == this.Rows && m.Cols == this.Cols, "Mat-Sub");
        var v = newFloatArray(this.Rows * this.Cols);
        for (var r = 0; r < this.Rows; r++) {
            for (var c = 0; c < this.Cols; c++) {
                var k = r * this.Cols + c;
                v[k] = this.dt[k] - m.dt[k];
            }
        }

        return new Mat(this.Rows, this.Cols, v);
    }

    Mul(m) {
        if (m instanceof Number) {

            return new Mat(this.Rows, this.Cols, this.dt.map(x => x * m));
        }
        Assert(m instanceof Mat && m.Rows == this.Rows && m.Cols == this.Cols && m.columnMajor == this.columnMajor, "Mat-Mul");
        var v = newFloatArray(this.Rows * this.Cols);
        for (var r = 0; r < this.Rows; r++) {
            for (var c = 0; c < this.Cols; c++) {
                var k = r * this.Cols + c;
                v[k] = this.dt[k] * m.dt[k];
            }
        }

        return new Mat(this.Rows, this.Cols, v);
    }

    Abs() {
        var v = newFloatArray(this.Rows * this.Cols);
        for (var r = 0; r < this.Rows; r++) {
            for (var c = 0; c < this.Cols; c++) {
                var k = r * this.Cols + c;
                v[k] = Math.abs(this.dt[k]);
            }
        }

        return new Mat(this.Rows, this.Cols, v);
    }

    Sum() {
        var sum = 0;
        for (var r = 0; r < this.Rows; r++) {
            for (var c = 0; c < this.Cols; c++) {
                sum += this.dt[r * this.Cols + c];
            }
        }

        return sum;
    }

    Dot(m) {
        Assert(m instanceof Mat && m.Rows == this.Cols, "Mat-Dot");

        var v = newFloatArray(this.Rows * m.Cols);
        for (var r = 0; r < this.Rows; r++) {
            for (var c = 0; c < m.Cols; c++) {
                var sum = 0;
                for (var k = 0; k < this.Cols; k++) {
                    sum += this.dt[r * this.Cols + k] * m.dt[k * m.Cols + c];
                }
                v[r * m.Cols + c] = sum;
            }
        }
        return new Mat(this.Rows, m.Cols, v);
    }

    toString() {
        var s = "[";
        for (var r = 0; r < this.Rows; r++) {
            if (r == 0) {

                s = s + " [";
            }
            else {

                s = s + "\r\n, [";
            }

            for (var c = 0; c < this.Cols; c++) {
                if (c != 0) {

                    s = s + ", ";
                }

                s = s + this.dt[r * this.Cols + c].toFixed(7);
            }

            s = s + "]";
        }

        s = s + " ]";

        return s;
    }
}
