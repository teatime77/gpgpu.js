class Mat {
    constructor(rows, cols, init, column_major) {
        if (!Mat.prototype.WebGL) {

            console.log("init WebGL");

            Mat.prototype.Prg = {};

            // -- Init Canvas
            var canvas = document.createElement('canvas');
            canvas.width = 32;
            canvas.height = 32;
            document.body.appendChild(canvas);

            // -- Init WebGL Context
            var gl = canvas.getContext('webgl2', { antialias: false });
            var isWebGL2 = !!gl;
            if (!isWebGL2) {
                console.log("WebGL 2 is not available. See How to get a WebGL 2 implementation");
                console.log("https://www.khronos.org/webgl/wiki/Getting_a_WebGL_Implementation");

                throw "WebGL 2 is not available.";
            }

            Mat.prototype.WebGL = gl;

            var shader = {};

            shader["vs-Texture"] = `#version 300 es

                precision highp float;
                precision highp int;

                uniform int B_Cols;

                uniform sampler2D A_Tex;
                uniform sampler2D B_Tex;

                layout(location = 0) in float idx_f;

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

            shader["vs-Uniform"] = `#version 300 es

                precision highp float;
                precision highp int;

                uniform int B_Cols;

                uniform vec4 A[_A_len_];
                uniform vec4 B[_B_len_];

                layout(location = 0) in float idx_f;

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

            shader["fs-transform"] = `#version 300 es
                precision highp float;
                precision highp int;

                out vec4 color;

                void main()
                {
                    color = vec4(1.0);
                }`;

            Mat.prototype.Shader = shader;
        }

        this.Rows = rows;
        this.Cols = cols;
        this.shape = [rows, cols];
        this.columnMajor = (column_major == undefined ? false : column_major);

        if (init) {

            Assert(init instanceof Float32Array && init.length == rows * cols, "Mat-init");
            this.dt = init;
        }
        else {

            this.dt = new Float32Array(rows * cols);
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

    T() {
        var v = new Float32Array(this.Cols * this.Rows);
        for (var r = 0; r < this.Cols; r++) {
            for (var c = 0; c < this.Rows; c++) {
                v[r * this.Rows + c] = this.dt[c * this.Cols + r];
            }
        }

        return new Mat(this.Cols, this.Rows, v);
    }

    transpose() {
        return this.T();
    }

    At(r, c) {
        return this.dt[r * this.Cols + c];
    }

    Set(r, c, val) {
        this.dt[r * this.Cols + c] = val;
    }

    Col(c) {
        var v = new Float32Array(this.Rows);
        for (var r = 0; r < this.Rows; r++) {
            v[r] = this.dt[r * this.Cols + c];
        }

        return new Mat(this.Rows, 1, v);
    }

    Add(m) {
        Assert(m instanceof Mat && m.Rows == this.Rows && m.Cols == this.Cols, "Mat-add");
        var v = new Float32Array(this.Rows * this.Cols);
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
        var v = new Float32Array(this.Rows * this.Cols);
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
        var v = new Float32Array(this.Rows * this.Cols);
        for (var r = 0; r < this.Rows; r++) {
            for (var c = 0; c < this.Cols; c++) {
                var k = r * this.Cols + c;
                v[k] = this.dt[k] - m.dt[r];
            }
        }

        return new Mat(this.Rows, this.Cols, v);
    }

    reduce(f) {
        var v = new Float32Array(this.Rows);
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
        var v = new Float32Array(this.Rows * this.Cols);
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
        var v = new Float32Array(this.Rows * this.Cols);
        for (var r = 0; r < this.Rows; r++) {
            for (var c = 0; c < this.Cols; c++) {
                var k = r * this.Cols + c;
                v[k] = this.dt[k] * m.dt[k];
            }
        }

        return new Mat(this.Rows, this.Cols, v);
    }

    Abs() {
        var v = new Float32Array(this.Rows * this.Cols);
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

        var v = new Float32Array(this.Rows * m.Cols);
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

    MakeProgram(gl, vshaderTransform, fshaderTransform) {
        var prg = gl.createProgram();
        gl.attachShader(prg, vshaderTransform);
        gl.attachShader(prg, fshaderTransform);

        var varyings = ['dot_val'];
        gl.transformFeedbackVaryings(prg, varyings, gl.SEPARATE_ATTRIBS);   // gl.INTERLEAVED_ATTRIBS
        gl.linkProgram(prg);

        // check
        var msg = gl.getProgramInfoLog(prg);
        if (msg) {
            console.log(msg);
        }

        msg = gl.getShaderInfoLog(vshaderTransform);
        if (msg) {
            console.log(msg);
        }

        gl.deleteShader(vshaderTransform);
        gl.deleteShader(fshaderTransform);

        return prg;
    }

    MakeFloat32Array(n) {
        var v = new Float32Array(n);

        for (var k = 0; k < n; k++) {
            v[k] = k;
        }

        return v;
    }

    MakeIdxBuffer(gl, gpu, element_count) {
        var buffer_2 = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer_2);
        gpu.vidx = this.MakeFloat32Array(element_count);
        for (var i = 0; i < element_count; i++) {
            gpu.vidx[i] = i;
        }
        gl.bufferData(gl.ARRAY_BUFFER, gpu.vidx, gl.STATIC_DRAW);
//        gl.vertexAttribPointer(0, 1, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, null);

        return buffer_2;
    }

    MakeShader(gl, type, source) {
        var shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);

        return shader;
    }

    MakeTex(gl, m, tex_id) {
        //            console.log("make tex : " + m.Cols / 4 + " " + m.Rows + "\r\n" + m.toString());

        var texture = gl.createTexture();

        gl.activeTexture(tex_id);
        gl.bindTexture(gl.TEXTURE_2D, texture);

        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

        return texture;
    }

    SetTex(gl, m, tex_id, texture) {
        gl.activeTexture(tex_id);
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, m.Cols / 4, m.Rows, 0, gl.RGBA, gl.FLOAT, m.dt);
    }

    Calc(B, use_tex) {
        var A = this;
        var vs_id;

        if (use_tex) {
            vs_id = "vs-Texture";
        }
        else {

            vs_id = "vs-Uniform";
        }
        var gl = Mat.prototype.WebGL;

        var element_count = A.Rows * B.Cols;

        var key = vs_id + ":" + A.Rows + "," + A.Cols + "," + B.Rows + "," + B.Cols;
        var gpu = Mat.prototype.Prg[key];
        if (!gpu) {

            gpu = { "key":key };
            Mat.prototype.Prg[key] = gpu;
            console.log("make gpu:" + gpu.key);

            var A_len = (A.Rows * A.Cols / 4).toString();
            var B_len = (B.Rows * B.Cols / 4).toString();
            var repeat = (A.Cols / 4).toString();
    //        console.log("A_len:[" + A_len + "] B_len:[" + B_len + "] repeat:[" + repeat + "]");
            gpu.buf_out_len = element_count * Float32Array.BYTES_PER_ELEMENT;

            var vsrc = Mat.prototype.Shader[vs_id].replace(/_repeat_/g, repeat).replace(/_A_len_/g, A_len).replace(/_B_len_/g, B_len);

            var fsrc = Mat.prototype.Shader['fs-transform'];
            var vshader = this.MakeShader(gl, gl.VERTEX_SHADER, vsrc);
            var fshader = this.MakeShader(gl, gl.FRAGMENT_SHADER, fsrc);
            gpu.program = this.MakeProgram(gl, vshader, fshader);
            gl.useProgram(gpu.program);

            gpu.loc_B_Cols = gl.getUniformLocation(gpu.program, 'B_Cols');

            gpu.buffer_2 = this.MakeIdxBuffer(gl, gpu, element_count);
            gpu.array_buffer = new ArrayBuffer(gpu.buf_out_len);

            if (use_tex){
                // テクスチャを使う場合

                gpu.loc_A_Tex = gl.getUniformLocation(gpu.program, 'A_Tex');
                gpu.loc_B_Tex = gl.getUniformLocation(gpu.program, 'B_Tex');

                // テクスチャの初期処理
                gpu.A_tex = this.MakeTex(gl, this, gl.TEXTURE0);
                gpu.B_tex = this.MakeTex(gl, B.T(), gl.TEXTURE1);

                console.log("loc:" + gpu.loc_B_Cols + ", " + gpu.loc_A_Tex + ", " + gpu.loc_B_Tex + " tex:" + gpu.A_tex + ", " + gpu.B_tex);
            }
            else{
                // ユニフォーム行列を使う場合

                gpu.loc_A = gl.getUniformLocation(gpu.program, 'A');
                gpu.loc_B = gl.getUniformLocation(gpu.program, 'B');

                console.log("loc:" + gpu.loc_B_Cols + ", " + gpu.loc_A + ", " + gpu.loc_B);
            }
        }
        else {

            gl.useProgram(gpu.program);
        }

        // -- Init Buffer

        // Feedback empty buffer
        var buf_out = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buf_out);
        gl.bufferData(gl.ARRAY_BUFFER, gpu.buf_out_len, gl.STATIC_COPY);
        gl.bindBuffer(gl.ARRAY_BUFFER, null);

//        gpu.buffer_2 = this.MakeIdxBuffer(gl, gpu, element_count);

        gl.bindBuffer(gl.ARRAY_BUFFER, gpu.buffer_2);
//        gl.bufferData(gl.ARRAY_BUFFER, gpu.vidx, gl.STATIC_DRAW);
        gl.vertexAttribPointer(0, 1, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(0);

        // テクスチャの値のセット
        if (use_tex) {

            this.SetTex(gl, this, gl.TEXTURE0 , gpu.A_tex);
            this.SetTex(gl, B.T(), gl.TEXTURE1, gpu.B_tex);
        }

        // -- Init TransformFeedback 
        var transformFeedback = gl.createTransformFeedback();
        gl.enable(gl.RASTERIZER_DISCARD);

        gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, transformFeedback);

        gl.useProgram(gpu.program);

        var tmp, startTime = new Date();
        tmp = new Date(); var t1 = tmp - startTime; startTime = tmp;

        // ユニフォーム変数の設定
        gl.uniform1i(gpu.loc_B_Cols, B.Cols);

        if (use_tex) {
            // テクスチャを使う場合

            gl.uniform1i(gpu.loc_A_Tex, 0);
            gl.uniform1i(gpu.loc_B_Tex, 1);
        }
        else {
            // ユニフォーム行列を使う場合

            gl.uniform4fv(gpu.loc_A, new Float32Array(this.dt));
            gl.uniform4fv(gpu.loc_B, new Float32Array(B.T().dt));
        }

        gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, buf_out);

        // 計算開始
        gl.beginTransformFeedback(gl.POINTS);    // TRIANGLES
        gl.drawArrays(gl.POINTS, 0, element_count);
        gl.endTransformFeedback();

        gl.disable(gl.RASTERIZER_DISCARD);
        gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, null);

        // 処理結果を表示
        gl.bindBuffer(gl.ARRAY_BUFFER, buf_out);

        gl.getBufferSubData(gl.ARRAY_BUFFER, 0, gpu.array_buffer);

        console.log("CALC:" + (new Date() - startTime) + "ms");

        var C = new Mat(A.Rows, B.Cols, new Float32Array(gpu.array_buffer));

        gl.bindBuffer(gl.ARRAY_BUFFER, null);

        // 終了処理
        gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null);//++
        gl.deleteTransformFeedback(transformFeedback);

        gl.deleteBuffer(buf_out);

        gl.useProgram(null);

        return C;
    }
}

Mat.prototype.Clear = function () {
    var gl = Mat.prototype.WebGL;
    for (key in Mat.prototype.Prg) {
        var gpu = Mat.prototype.Prg[key];

        gl.bindBuffer(gl.ARRAY_BUFFER, null);
        gl.deleteBuffer(gpu.buffer_2);

        if(gpu.A_tex){

            gl.bindTexture(gl.TEXTURE_2D, null);
            gl.deleteTexture(gpu.A_tex);
            gl.deleteTexture(gpu.B_tex);
        }

        gl.deleteProgram(gpu.program);
        console.log("clear gpu:" + gpu.key);
    }
}
