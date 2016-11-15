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

    MakeProgram(gl, vshaderTransform, fshaderTransform, varyings) {
        var prg = gl.createProgram();
        gl.attachShader(prg, vshaderTransform);
        gl.attachShader(prg, fshaderTransform);

        gl.transformFeedbackVaryings(prg, varyings, gl.INTERLEAVED_ATTRIBS);   //  gl.SEPARATE_ATTRIBS
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
        var idx_buffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, idx_buffer);
        gpu.vidx = this.MakeFloat32Array(element_count);
        for (var i = 0; i < element_count; i++) {
            gpu.vidx[i] = i;
        }
        gl.bufferData(gl.ARRAY_BUFFER, gpu.vidx, gl.STATIC_DRAW);
        gl.bindBuffer(gl.ARRAY_BUFFER, null);

        return idx_buffer;
    }

    MakeShader(gl, type, source) {
        var shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);

        return shader;
    }

    MakeTex(gl, tex_id) {
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

    Calc(param) {
        var gl = Mat.prototype.WebGL;

        var TEXTUREs = [gl.TEXTURE0, gl.TEXTURE1, gl.TEXTURE2, gl.TEXTURE3];

        var gpu = Mat.prototype.Prg[param.key];
        if (!gpu) {

            gpu = {};
            Mat.prototype.Prg[param.key] = gpu;

            gpu.key = param.key;

            gpu.outBufferSize = param.elementCount * Float32Array.BYTES_PER_ELEMENT;

            var fsrc = Mat.prototype.Shader['fs-transform'];
            var vshader = this.MakeShader(gl, gl.VERTEX_SHADER, param.vsrc);
            var fshader = this.MakeShader(gl, gl.FRAGMENT_SHADER, fsrc);
            gpu.program = this.MakeProgram(gl, vshader, fshader, param.varyings);
            gl.useProgram(gpu.program);

            // ユニフォーム変数の初期処理
            gpu.locUniforms = [];
            for(let u of param.uniforms) {

                var loc = gl.getUniformLocation(gpu.program, u.name);
                gpu.locUniforms.push(loc);
            }

            // テクスチャの初期処理
            gpu.locTextures = [];
            gpu.Textures = [];
            for (var i = 0; i < param.textures.length; i++) {

                var loc = gl.getUniformLocation(gpu.program, param.textures[i].name);
                gpu.locTextures.push(loc);

                var tex = this.MakeTex(gl, TEXTUREs[i]);
                gpu.Textures.push(tex);
            }

            gpu.idxBuffer = this.MakeIdxBuffer(gl, gpu, param.elementCount);
            gpu.arrayBuffer = new ArrayBuffer(gpu.outBufferSize);

            // Feedback empty buffer
            gpu.outBuffer = gl.createBuffer();
            gl.bindBuffer(gl.ARRAY_BUFFER, gpu.outBuffer);
            gl.bufferData(gl.ARRAY_BUFFER, gpu.outBufferSize, gl.STATIC_COPY);
            gl.bindBuffer(gl.ARRAY_BUFFER, null);

            // -- Init TransformFeedback 
            gpu.transformFeedback = gl.createTransformFeedback();
        }
        else {

            gl.useProgram(gpu.program);
        }

        // -- Init Buffer

        gl.bindBuffer(gl.ARRAY_BUFFER, gpu.idxBuffer);
        gl.vertexAttribPointer(0, 1, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(0);

        gl.useProgram(gpu.program);

        gl.enable(gl.RASTERIZER_DISCARD);

        gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, gpu.transformFeedback);

        // テクスチャの値のセット
        for (var i = 0; i < param.textures.length; i++) {

            this.SetTex(gl, param.textures[i].value, TEXTUREs[i], gpu.Textures[i]);
            gl.uniform1i(gpu.locTextures[i], i);
        }

        // ユニフォーム変数のセット
        for (var i = 0; i < param.uniforms.length; i++) {
            var u = param.uniforms[i];
            if (u.value instanceof Mat) {

                gl.uniform4fv(gpu.locUniforms[i], new Float32Array(u.value.dt));
            }
            else {

                gl.uniform1i(gpu.locUniforms[i], u.value);
            }
        }

        gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, gpu.outBuffer);

        // 計算開始
        gl.beginTransformFeedback(gl.POINTS);    // TRIANGLES
        gl.drawArrays(gl.POINTS, 0, param.elementCount);
        gl.endTransformFeedback();

        gl.disable(gl.RASTERIZER_DISCARD);
        gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, null);

        // 処理結果を表示
        gl.bindBuffer(gl.ARRAY_BUFFER, gpu.outBuffer);

        gl.getBufferSubData(gl.ARRAY_BUFFER, 0, gpu.arrayBuffer);

        var ret = new Float32Array(gpu.arrayBuffer);

        gl.bindBuffer(gl.ARRAY_BUFFER, null);

        // 終了処理
        gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null);

        gl.useProgram(null);

        return ret;
    }
}

Mat.prototype.Clear = function () {
    var gl = Mat.prototype.WebGL;
    for (key in Mat.prototype.Prg) {
        var gpu = Mat.prototype.Prg[key];

        gl.bindBuffer(gl.ARRAY_BUFFER, null);
        gl.deleteBuffer(gpu.idxBuffer);
        gl.deleteBuffer(gpu.outBuffer);
        gl.deleteTransformFeedback(gpu.transformFeedback);

        gl.bindTexture(gl.TEXTURE_2D, null);
        for(let tex of gpu.Textures) {

            gl.deleteTexture(tex);
        }

        gl.deleteProgram(gpu.program);
        console.log("clear gpu:" + gpu.key);
    }
}
