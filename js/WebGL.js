// JavaScript source code

function CreateWebGLLib() {
    let gl;

    function chk() {
        Assert(gl.getError() == gl.NO_ERROR);
    }

    class WebGLLib {

        constructor() {
            console.log("init WebGL");

            this.packages = {};

            // -- Init Canvas
            var canvas = document.createElement('canvas');
            canvas.width = 32;
            canvas.height = 32;
            document.body.appendChild(canvas);

            // -- Init WebGL Context
            gl = canvas.getContext('webgl2', { antialias: false });
            var isWebGL2 = !!gl;
            if (!isWebGL2) {
                console.log("WebGL 2 is not available. See How to get a WebGL 2 implementation");
                console.log("https://www.khronos.org/webgl/wiki/Getting_a_WebGL_Implementation");

                throw "WebGL 2 is not available.";
            }

            this.TEXTUREs = [gl.TEXTURE0, gl.TEXTURE1, gl.TEXTURE2, gl.TEXTURE3];
        }

        WebGLClear() {
            for (var key in this.packages) {
                var pkg = this.packages[key];

                gl.bindBuffer(gl.ARRAY_BUFFER, null); chk();
                gl.deleteBuffer(pkg.idxBuffer); chk();
                for(let buf of pkg.feedbackBuffers) {
                    gl.deleteBuffer(buf); chk();
                }
                gl.deleteTransformFeedback(pkg.transformFeedback); chk();

                gl.bindTexture(gl.TEXTURE_2D, null); chk();
                gl.bindTexture(gl.TEXTURE_3D, null); chk();
                for(let tex of pkg.Textures) {

                    gl.deleteTexture(tex); chk();
                }

                gl.deleteProgram(pkg.program); chk();
                console.log("clear pkg:" + pkg.key);
            }
        }

        makeProgram(vshaderTransform, fshaderTransform, varyings) {
            var prg = gl.createProgram(); chk();
            gl.attachShader(prg, vshaderTransform); chk();
            gl.attachShader(prg, fshaderTransform); chk();

            var varying_names = varyings.map(x => x.name);
            gl.transformFeedbackVaryings(prg, varying_names, gl.SEPARATE_ATTRIBS); chk();   // gl.INTERLEAVED_ATTRIBS 
            gl.linkProgram(prg); chk();

            // check
            var msg = gl.getProgramInfoLog(prg); chk();
            if (msg) {
                console.log(msg);
            }

            msg = gl.getShaderInfoLog(vshaderTransform); chk();
            if (msg) {
                console.log(msg);
            }

            gl.deleteShader(vshaderTransform); chk();
            gl.deleteShader(fshaderTransform); chk();

            return prg;
        }

        MakeIdxBuffer(pkg, element_count) {
            pkg.idxBuffer = gl.createBuffer(); chk();
            gl.bindBuffer(gl.ARRAY_BUFFER, pkg.idxBuffer); chk();

            pkg.vidx = new Float32Array(element_count);
            for (var i = 0; i < element_count; i++) {
                pkg.vidx[i] = i;
            }
            gl.bufferData(gl.ARRAY_BUFFER, pkg.vidx, gl.STATIC_DRAW); chk();
            gl.bindBuffer(gl.ARRAY_BUFFER, null); chk();
        }

        makeShader(type, source) {
            var shader = gl.createShader(type); chk();
            gl.shaderSource(shader, source); chk();
            gl.compileShader(shader); chk();

            return shader;
        }

        makeTexture(tex_id, texture_value) {
            var dim = texture_value.shape.length == 3 ? gl.TEXTURE_3D : gl.TEXTURE_2D;

            var texture = gl.createTexture(); chk();

            gl.activeTexture(tex_id); chk();
            gl.bindTexture(dim, texture); chk();

            gl.texParameteri(dim, gl.TEXTURE_MAG_FILTER, gl.NEAREST); chk();
            gl.texParameteri(dim, gl.TEXTURE_MIN_FILTER, gl.NEAREST); chk();
            gl.texParameteri(dim, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); chk();
            gl.texParameteri(dim, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE); chk();

            return texture;
        }

        setTextureData(texture_value, tex_id, texture) {
            var dim = texture_value.shape.length == 3 ? gl.TEXTURE_3D : gl.TEXTURE_2D;

            gl.activeTexture(tex_id); chk();
            gl.bindTexture(dim, texture); chk();
            if (dim == gl.TEXTURE_2D) {

                gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, texture_value.Cols / 4, texture_value.Rows, 0, gl.RGBA, gl.FLOAT, texture_value.dt); chk();
            }
            else {
                Assert(dim == gl.TEXTURE_3D, "Set-Tex");

                gl.texImage3D(gl.TEXTURE_3D, 0, gl.RGBA32F, texture_value.Cols / 4, texture_value.Rows, texture_value.Depth, 0, gl.RGBA, gl.FLOAT, texture_value.dt); chk();
            }
        }

        makePackage(param) {
            var pkg = {};
            this.packages[param.key] = pkg;

            pkg.key = param.key;

            var fsrc = Shaders['fs-transform'];
            var vertex_shader = this.makeShader(gl.VERTEX_SHADER, param.shaderText);

            var fragment_shader = this.makeShader(gl.FRAGMENT_SHADER, fsrc);

            pkg.program = this.makeProgram(vertex_shader, fragment_shader, param.varyings);
            gl.useProgram(pkg.program); chk();

            // ユニフォーム変数の初期処理
            pkg.locUniforms = [];
            for(let u of param.uniforms) {

                var loc = gl.getUniformLocation(pkg.program, u.name); chk();
                pkg.locUniforms.push(loc);
            }

            // テクスチャの初期処理
            pkg.locTextures = [];
            pkg.Textures = [];
            for (var i = 0; i < param.textures.length; i++) {

                var loc = gl.getUniformLocation(pkg.program, param.textures[i].name); chk();
                pkg.locTextures.push(loc);

                var tex = this.makeTexture(this.TEXTUREs[i], param.textures[i].value);
                pkg.Textures.push(tex);
            }

            if (param.attributes) {

                pkg.AttribBuffers = [];
                for (var i = 0; i < param.attributes.length; i++) {
                    var vbo = gl.createBuffer();
//                    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);

                    pkg.AttribBuffers.push(vbo);
                }
            }
            else {

                this.MakeIdxBuffer(pkg, param.elementCount);
            }

            pkg.feedbackBuffers = [];

            var out_buffer_size = param.elementDim * param.elementCount * Float32Array.BYTES_PER_ELEMENT;

            for (let varying of param.varyings) {

                // Feedback empty buffer
                var buf = gl.createBuffer(); chk();
                gl.bindBuffer(gl.ARRAY_BUFFER, buf); chk();
                gl.bufferData(gl.ARRAY_BUFFER, out_buffer_size, gl.STATIC_COPY); chk();
                gl.bindBuffer(gl.ARRAY_BUFFER, null); chk();

                pkg.feedbackBuffers.push(buf);
            }

            // -- Init TransformFeedback 
            pkg.transformFeedback = gl.createTransformFeedback(); chk();

            return pkg;
        }

        compute(param) {
            var pkg = this.packages[param.key];
            if (!pkg) {

                pkg = this.makePackage(param);
            }
            else {

                gl.useProgram(pkg.program); chk();
            }

            // -- Init Buffer

            if (param.attributes) {

                for (var i = 0; i < param.attributes.length; i++) {
                    var attrib = param.attributes[i];

                    gl.bindBuffer(gl.ARRAY_BUFFER, pkg.AttribBuffers[i]); chk();
                    gl.vertexAttribPointer(i, 1, gl.FLOAT, false, 4, 0); chk();
                    gl.enableVertexAttribArray(i); chk();
                    gl.bindAttribLocation(pkg.program, i, attrib.name);
                    gl.bufferData(gl.ARRAY_BUFFER, attrib.value, gl.STATIC_DRAW);
                }
            }
            else {

                gl.bindBuffer(gl.ARRAY_BUFFER, pkg.idxBuffer); chk();
                gl.vertexAttribPointer(0, 1, gl.FLOAT, false, 0, 0); chk();
                gl.enableVertexAttribArray(0); chk();
            }

            gl.useProgram(pkg.program); chk();

            gl.enable(gl.RASTERIZER_DISCARD); chk();

            gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, pkg.transformFeedback); chk();

            // テクスチャの値のセット
            for (var i = 0; i < param.textures.length; i++) {

                this.setTextureData(param.textures[i].value, this.TEXTUREs[i], pkg.Textures[i]);
                gl.uniform1i(pkg.locTextures[i], i); chk();
            }

            // ユニフォーム変数のセット
            for (var i = 0; i < param.uniforms.length; i++) {
                var u = param.uniforms[i];
                if (u.value instanceof Mat) {

                    if (u.type == "vec4") {

    //                    gl.uniform4fv(pkg.locUniforms[i], new Float32Array(u.value.dt)); chk();
                        gl.uniform4fv(pkg.locUniforms[i], u.value.dt); chk();
                    }
                    else {
                        gl.uniform1fv(pkg.locUniforms[i], u.value.dt); chk();
                    }
                }
                else if (u.value instanceof Float32Array) {

    //                gl.uniform1fv(pkg.locUniforms[i], new Float32Array(u.value)); chk();
                    gl.uniform1fv(pkg.locUniforms[i], u.value); chk();
                }
                else {

                    gl.uniform1i(pkg.locUniforms[i], u.value); chk();
                }
            }

            for (var i = 0; i < param.varyings.length; i++) {

                gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, i, pkg.feedbackBuffers[i]); chk();
            }

            // 計算開始
            gl.beginTransformFeedback(gl.POINTS); chk();    // TRIANGLES
            gl.drawArrays(gl.POINTS, 0, param.elementCount); chk();
            gl.endTransformFeedback(); chk();

            gl.disable(gl.RASTERIZER_DISCARD); chk();

            for (var i = 0; i < param.varyings.length; i++) {

                gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, i, null); chk();

                // 処理結果を表示
                gl.bindBuffer(gl.ARRAY_BUFFER, pkg.feedbackBuffers[i]); chk();

                var out_buf = param.varyings[i].value;
                if (out_buf instanceof Mat) {
                    out_buf = out_buf.dt;
                }

                gl.getBufferSubData(gl.ARRAY_BUFFER, 0, out_buf); chk();

                gl.bindBuffer(gl.ARRAY_BUFFER, null); chk();
            }

            // 終了処理
            gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null); chk();

            gl.useProgram(null); chk();
        }
    }

    return new WebGLLib();
}
