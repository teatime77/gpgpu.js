// JavaScript source code

function gg() {
    Assert(WebGL2.GL.getError() == WebGL2.GL.NO_ERROR);
}

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

            gl.bindBuffer(gl.ARRAY_BUFFER, null); gg();
            gl.deleteBuffer(gpu.idxBuffer); gg();
            for(let buf of gpu.outBuffers) {
                gl.deleteBuffer(buf); gg();
            }
            gl.deleteTransformFeedback(gpu.transformFeedback); gg();

            gl.bindTexture(gl.TEXTURE_2D, null); gg();
            gl.bindTexture(gl.TEXTURE_3D, null); gg();
            for(let tex of gpu.Textures) {

                gl.deleteTexture(tex); gg();
            }

            gl.deleteProgram(gpu.program); gg();
            console.log("clear gpu:" + gpu.key);
        }
    }

    MakeProgram(gl, vshaderTransform, fshaderTransform, varyings) {
        var prg = gl.createProgram(); gg();
        gl.attachShader(prg, vshaderTransform); gg();
        gl.attachShader(prg, fshaderTransform); gg();

        gl.transformFeedbackVaryings(prg, varyings, gl.SEPARATE_ATTRIBS); gg();   // gl.INTERLEAVED_ATTRIBS 
        gl.linkProgram(prg); gg();

        // check
        var msg = gl.getProgramInfoLog(prg); gg();
        if (msg) {
            console.log(msg);
        }

        msg = gl.getShaderInfoLog(vshaderTransform); gg();
        if (msg) {
            console.log(msg);
        }

        gl.deleteShader(vshaderTransform); gg();
        gl.deleteShader(fshaderTransform); gg();

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
        var idx_buffer = gl.createBuffer(); gg();
        gl.bindBuffer(gl.ARRAY_BUFFER, idx_buffer); gg();
        gpu.vidx = this.MakeFloat32Array(element_count);
        for (var i = 0; i < element_count; i++) {
            gpu.vidx[i] = i;
        }
        gl.bufferData(gl.ARRAY_BUFFER, gpu.vidx, gl.STATIC_DRAW); gg();
        gl.bindBuffer(gl.ARRAY_BUFFER, null); gg();

        return idx_buffer;
    }

    MakeShader(gl, type, source) {
        var shader = gl.createShader(type); gg();
        gl.shaderSource(shader, source); gg();
        gl.compileShader(shader); gg();

        return shader;
    }

    MakeTex(gl, tex_id, dim) {
        var texture = gl.createTexture(); gg();

        gl.activeTexture(tex_id); gg();
        gl.bindTexture(dim, texture); gg();

        gl.texParameteri(dim, gl.TEXTURE_MAG_FILTER, gl.NEAREST); gg();
        gl.texParameteri(dim, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gg();
        gl.texParameteri(dim, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gg();
        gl.texParameteri(dim, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE); gg();

        return texture;
    }

    SetTex(gl, m, tex_id, dim, texture) {
        gl.activeTexture(tex_id); gg();
        gl.bindTexture(dim, texture); gg();
        if (dim == gl.TEXTURE_2D) {

            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, m.Cols / 4, m.Rows, 0, gl.RGBA, gl.FLOAT, m.dt); gg();
        }
        else {
            Assert(dim == gl.TEXTURE_3D, "Set-Tex");

            gl.texImage3D(gl.TEXTURE_3D, 0, gl.RGBA32F, m.Cols / 4, m.Rows, m.Depth, 0, gl.RGBA, gl.FLOAT, m.dt); gg();
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
            gl.useProgram(gpu.program); gg();

            // ユニフォーム変数の初期処理
            gpu.locUniforms = [];
            for(let u of param.uniforms) {

                var loc = gl.getUniformLocation(gpu.program, u.name); gg();
                gpu.locUniforms.push(loc);
            }

            // テクスチャの初期処理
            gpu.locTextures = [];
            gpu.Textures = [];
            for (var i = 0; i < param.textures.length; i++) {

                var loc = gl.getUniformLocation(gpu.program, param.textures[i].name); gg();
                gpu.locTextures.push(loc);

                var tex = this.MakeTex(gl, TEXTUREs[i], param.textures[i].dim);
                gpu.Textures.push(tex);
            }

            gpu.idxBuffer = this.MakeIdxBuffer(gl, gpu, param.elementCount);
            gpu.outBuffers = [];

            var out_buffer_size = param.elementDim * param.elementCount * Float32Array.BYTES_PER_ELEMENT;
            if (param.arrayBuffers) {

                gpu.arrayBuffers = param.arrayBuffers;
            }
            else{

                gpu.arrayBuffers = xrange(param.varyings.length).map(x => new Float32Array(param.elementCount));    // new ArrayBuffer(out_buffer_size)
            }

            for (var i = 0; i < param.varyings.length; i++) {

                // Feedback empty buffer
                var buf = gl.createBuffer(); gg();
                gl.bindBuffer(gl.ARRAY_BUFFER, buf); gg();
                gl.bufferData(gl.ARRAY_BUFFER, out_buffer_size, gl.STATIC_COPY); gg();
                gl.bindBuffer(gl.ARRAY_BUFFER, null); gg();

                gpu.outBuffers.push(buf);
            }

            // -- Init TransformFeedback 
            gpu.transformFeedback = gl.createTransformFeedback(); gg();
        }
        else {

            gl.useProgram(gpu.program); gg();
        }

        // -- Init Buffer

        gl.bindBuffer(gl.ARRAY_BUFFER, gpu.idxBuffer); gg();
        gl.vertexAttribPointer(0, 1, gl.FLOAT, false, 0, 0); gg();
        gl.enableVertexAttribArray(0); gg();

        gl.useProgram(gpu.program); gg();

        gl.enable(gl.RASTERIZER_DISCARD); gg();

        gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, gpu.transformFeedback); gg();

        // テクスチャの値のセット
        for (var i = 0; i < param.textures.length; i++) {

            this.SetTex(gl, param.textures[i].value, TEXTUREs[i], param.textures[i].dim, gpu.Textures[i]);
            gl.uniform1i(gpu.locTextures[i], i); gg();
        }

        // ユニフォーム変数のセット
        for (var i = 0; i < param.uniforms.length; i++) {
            var u = param.uniforms[i];
            if (u.value instanceof Mat) {

                gl.uniform4fv(gpu.locUniforms[i], newFloatArray(u.value.dt)); gg();
            }
            else if (u.value instanceof Float32Array) {

//                gl.uniform1fv(gpu.locUniforms[i], newFloatArray(u.value)); gg();
                gl.uniform1fv(gpu.locUniforms[i], u.value); gg();
            }
            else {

                gl.uniform1i(gpu.locUniforms[i], u.value); gg();
            }
        }

        for (var i = 0; i < param.varyings.length; i++) {

            gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, i, gpu.outBuffers[i]); gg();
        }

        // 計算開始
        gl.beginTransformFeedback(gl.POINTS); gg();    // TRIANGLES
        gl.drawArrays(gl.POINTS, 0, param.elementCount); gg();
        gl.endTransformFeedback(); gg();

        gl.disable(gl.RASTERIZER_DISCARD); gg();

        var ret = [];
        for (var i = 0; i < param.varyings.length; i++) {

            gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, i, null); gg();

            // 処理結果を表示
            gl.bindBuffer(gl.ARRAY_BUFFER, gpu.outBuffers[i]); gg();

            gl.getBufferSubData(gl.ARRAY_BUFFER, 0, gpu.arrayBuffers[i]); gg();

            ret.push( newFloatArray(gpu.arrayBuffers[i]) );

            gl.bindBuffer(gl.ARRAY_BUFFER, null); gg();
        }

        // 終了処理
        gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null); gg();

        gl.useProgram(null); gg();

        return ret;
    }
}
