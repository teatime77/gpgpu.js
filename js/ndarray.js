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

        this.nElement = this.shape.reduce((x, y) => x * y);

        if (this.dt) {

            Assert(this.dt.length == this.nElement);
        }
        else {
            this.dt = new Float32Array(this.nElement);
        }
    }

    map(f) {
        return new ArrayView(this.nrow, this.ncol, this.dt.map(f));
    }

    T(m) {
        if (m == undefined) {
            m = new ArrayView(this.ncol, this.nrow);
        }
        for (var r = 0; r < this.ncol; r++) {
            for (var c = 0; c < this.nrow; c++) {
                m.dt[r * this.nrow + c] = this.dt[c * this.ncol + r];
            }
        }

        return m;
    }

    transpose() {
        return this.T();
    }

    At(r, c) {
        Assert(r < this.nrow && c < this.ncol, "ArrayView-at");
        return this.dt[r * this.ncol + c];
    }

    Set(r, c, val) {
        Assert(r < this.nrow && c < this.ncol, "ArrayView-set");

        this.dt[r * this.ncol + c] = val;
    }

    Set3(d, r, c, val) {
        Assert(d < this.shape[this.shape.length - 3] && r < this.nrow && c < this.ncol, "ArrayView-set3");

        this.dt[(d * this.nrow + r) * this.ncol + c] = val;
    }

    At3(d, r, c) {
        Assert(d < this.shape[this.shape.length - 3] && r < this.nrow && c < this.ncol, "ArrayView-at3");

        return this.dt[(d * this.nrow + r) * this.ncol + c];
    }

    Col(c) {
        var v = new Float32Array(this.nrow);
        for (var r = 0; r < this.nrow; r++) {
            v[r] = this.dt[r * this.ncol + c];
        }

        return new ArrayView(this.nrow, 1, v);
    }

    CopyRows(m, r_src, r_dst, r_cnt) {
        Assert(m instanceof ArrayView && m.ncol == this.ncol && r_src + r_cnt <= this.nrow && r_dst + r_cnt <= m.nrow, "copy-rows");

        for (var r = 0; r < r_cnt; r++) {
            for (var c = 0; c < this.ncol; c++) {
                m.dt[(r_dst + r) * m.ncol + c] = this.dt[(r_src + r) * this.ncol + c];
            }
        }
    }

    CopyCols(m, c_src, c_dst, c_cnt) {
        Assert(m instanceof ArrayView && m.nrow == this.nrow && c_src + c_cnt <= this.ncol && c_dst + c_cnt <= m.ncol, "copy-cols");

        for (var r = 0; r < this.nrow; r++) {
            for (var c = 0; c < c_cnt; c++) {
                m.dt[ r * m.ncol + c_dst + c ] = this.dt[ r * this.ncol + c_src + c ];
            }
        }
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

    AddV(m) {
        Assert(m instanceof ArrayView && m.nrow == this.nrow && m.ncol == 1, "ArrayView-add-V");
        var v = new Float32Array(this.nrow * this.ncol);
        for (var r = 0; r < this.nrow; r++) {
            for (var c = 0; c < this.ncol; c++) {
                var k = r * this.ncol + c;
                v[k] = this.dt[k] + m.dt[r];
            }
        }

        return new ArrayView(this.nrow, this.ncol, v);
    }

    SubV(m) {
        Assert(m instanceof ArrayView && m.nrow == this.nrow && m.ncol == 1, "ArrayView-sub-V");
        var v = new Float32Array(this.nrow * this.ncol);
        for (var r = 0; r < this.nrow; r++) {
            for (var c = 0; c < this.ncol; c++) {
                var k = r * this.ncol + c;
                v[k] = this.dt[k] - m.dt[r];
            }
        }

        return new ArrayView(this.nrow, this.ncol, v);
    }

    reduce(f) {
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
        Assert(m instanceof ArrayView && m.nrow == this.nrow && m.ncol == this.ncol && m.columnMajor == this.columnMajor, "ArrayView-Mul");
        var v = new Float32Array(this.nrow * this.ncol);
        for (var r = 0; r < this.nrow; r++) {
            for (var c = 0; c < this.ncol; c++) {
                var k = r * this.ncol + c;
                v[k] = this.dt[k] * m.dt[k];
            }
        }

        return new ArrayView(this.nrow, this.ncol, v);
    }

    Abs() {
        var v = new Float32Array(this.nrow * this.ncol);
        for (var r = 0; r < this.nrow; r++) {
            for (var c = 0; c < this.ncol; c++) {
                var k = r * this.ncol + c;
                v[k] = Math.abs(this.dt[k]);
            }
        }

        return new ArrayView(this.nrow, this.ncol, v);
    }

    Sum() {
        var sum = 0;
        for (var r = 0; r < this.nrow; r++) {
            for (var c = 0; c < this.ncol; c++) {
                sum += this.dt[r * this.ncol + c];
            }
        }

        return sum;
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

    toString() {
        var s = "[";
        for (var r = 0; r < this.nrow; r++) {
            if (r == 0) {

                s = s + " [";
            }
            else {

                s = s + "\r\n, [";
            }

            for (var c = 0; c < this.ncol; c++) {
                if (c != 0) {

                    s = s + ", ";
                }

                s = s + this.dt[r * this.ncol + c].toFixed(7);
            }

            s = s + "]";
        }

        s = s + " ]";

        return s;
    }
}
