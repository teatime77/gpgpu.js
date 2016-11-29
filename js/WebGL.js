// JavaScript source code

class WebGLLib {

    constructor() {
        console.log("init WebGL");

        this.Prg = {};

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

        this.GL = gl;
    }

    WebGLClear() {
        var gl = this.GL;
        for (var key in this.Prg) {
            var gpu = this.Prg[key];

            gl.bindBuffer(gl.ARRAY_BUFFER, null);
            gl.deleteBuffer(gpu.idxBuffer);
            for(let buf of gpu.outBuffers) {
                gl.deleteBuffer(buf);
            }
            gl.deleteTransformFeedback(gpu.transformFeedback);

            gl.bindTexture(gl.TEXTURE_2D, null);
            gl.bindTexture(gl.TEXTURE_3D, null);
            for(let tex of gpu.Textures) {

                gl.deleteTexture(tex);
            }

            gl.deleteProgram(gpu.program);
            console.log("clear gpu:" + gpu.key);
        }
    }

    MakeProgram(gl, vshaderTransform, fshaderTransform, varyings) {
        var prg = gl.createProgram();
        gl.attachShader(prg, vshaderTransform);
        gl.attachShader(prg, fshaderTransform);

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
        var v = newFloatArray(n);

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

    MakeTex(gl, tex_id, dim) {
        var texture = gl.createTexture();

        gl.activeTexture(tex_id);
        gl.bindTexture(dim, texture);

        gl.texParameteri(dim, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(dim, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(dim, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(dim, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

        return texture;
    }

    SetTex(gl, m, tex_id, dim, texture) {
        gl.activeTexture(tex_id);
        gl.bindTexture(dim, texture);
        if (dim == gl.TEXTURE_2D) {

            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, m.Cols / 4, m.Rows, 0, gl.RGBA, gl.FLOAT, m.dt);
        }
        else {
            Assert(dim == gl.TEXTURE_3D, "Set-Tex");

            gl.texImage3D(gl.TEXTURE_3D, 0, gl.RGBA32F, m.Cols / 4, m.Rows, m.Depth, 0, gl.RGBA, gl.FLOAT, m.dt);
        }
    }

    Calc(param) {
        var gl = this.GL;

        var TEXTUREs = [gl.TEXTURE0, gl.TEXTURE1, gl.TEXTURE2, gl.TEXTURE3];

        var gpu = this.Prg[param.key];
        if (!gpu) {

            gpu = {};
            this.Prg[param.key] = gpu;

            gpu.key = param.key;

            var fsrc = Shaders['fs-transform'];
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

                var tex = this.MakeTex(gl, TEXTUREs[i], param.textures[i].dim);
                gpu.Textures.push(tex);
            }

            gpu.idxBuffer = this.MakeIdxBuffer(gl, gpu, param.elementCount);
            gpu.outBuffers = [];

            var out_buffer_size = param.elementCount * Float32Array.BYTES_PER_ELEMENT;
            if (param.arrayBuffers) {

                gpu.arrayBuffers = param.arrayBuffers;
            }
            else{

                gpu.arrayBuffers = xrange(param.varyings.length).map(x => new ArrayBuffer(out_buffer_size));
            }

            for (var i = 0; i < param.varyings.length; i++) {

                // Feedback empty buffer
                var buf = gl.createBuffer();
                gl.bindBuffer(gl.ARRAY_BUFFER, buf);
                gl.bufferData(gl.ARRAY_BUFFER, out_buffer_size, gl.STATIC_COPY);
                gl.bindBuffer(gl.ARRAY_BUFFER, null);

                gpu.outBuffers.push(buf);
            }

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

            this.SetTex(gl, param.textures[i].value, TEXTUREs[i], param.textures[i].dim, gpu.Textures[i]);
            gl.uniform1i(gpu.locTextures[i], i);
        }

        // ユニフォーム変数のセット
        for (var i = 0; i < param.uniforms.length; i++) {
            var u = param.uniforms[i];
            if (u.value instanceof Mat) {

                gl.uniform4fv(gpu.locUniforms[i], newFloatArray(u.value.dt));
            }
            else if (u.value instanceof Float32Array) {

                gl.uniform1fv(gpu.locUniforms[i], newFloatArray(u.value));
            }
            else {

                gl.uniform1i(gpu.locUniforms[i], u.value);
            }
        }

        for (var i = 0; i < param.varyings.length; i++) {

            gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, i, gpu.outBuffers[i]);
        }

        // 計算開始
        gl.beginTransformFeedback(gl.POINTS);    // TRIANGLES
        gl.drawArrays(gl.POINTS, 0, param.elementCount);
        gl.endTransformFeedback();

        gl.disable(gl.RASTERIZER_DISCARD);

        var ret = [];
        for (var i = 0; i < param.varyings.length; i++) {

            gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, i, null);

            // 処理結果を表示
            gl.bindBuffer(gl.ARRAY_BUFFER, gpu.outBuffers[i]);

            gl.getBufferSubData(gl.ARRAY_BUFFER, 0, gpu.arrayBuffers[i]);

            ret.push( newFloatArray(gpu.arrayBuffers[i]) );

            gl.bindBuffer(gl.ARRAY_BUFFER, null);
        }

        // 終了処理
        gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null);

        gl.useProgram(null);

        return ret;
    }
}
