﻿class Mat {
    constructor(rows, cols, init, column_major, depth) {
        if(cols  == undefined){
            cols = 1;
        }
        this.Rows  = rows;
        this.Cols  = cols;
        this.Depth = (depth == undefined ? 1 : depth);
        this.shape = [rows, cols];
        this.columnMajor = (column_major == undefined ? false : column_major);

        Assert(!this.columnMajor);

        if (init) {

            Assert((init instanceof Float32Array || init instanceof Float64Array) && init.length == rows * cols * this.Depth, "Mat-init");
            this.dt = init;
        }
        else {

            this.dt = newFloatArray(rows * cols * this.Depth);
            /*
            for (var r = 0; r < rows; r++) {
                for (var c = 0; c < cols; c++) {
                    //                            this.dt[r * cols + c] = r * 1000 + c;
                    this.dt[r * cols + c] = Math.random();
                }
            }
            */
        }
    }

    map(f) {
        return new Mat(this.Rows, this.Cols, this.dt.map(f), this.columnMajor);
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
