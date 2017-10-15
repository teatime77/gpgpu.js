// JavaScript source code

function MakeFloat32Index(n) {
    var v = new Float32Array(n);
    for (var i = 0; i < n; i++) {
        v[i] = i;
    }

    return v;
}

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

        parseShader(pkg, param) {
            pkg.attributes = [];
            pkg.uniforms = [];
            pkg.textures = [];
            pkg.varyings = [];
            var lines = param.shaderText.split(/(\r\n|\r|\n)+/);
            for(let line of lines) {

                var tokens = line.split(/[\s\t]+/);
                if (tokens.length < 3) {
                    continue;
                }

                var tkn0 = tokens[0];
                var tkn1 = tokens[1];
                var tkn2 = tokens[2];

                if (tkn0 != "in" && tkn0 != "uniform" && tkn0 != "out") {
                    continue;
                }
                Assert(tkn1 == "int" || tkn1 == "float" || tkn1 == "vec2" || tkn1 == "vec3" || tkn1 == "vec4" || tkn1 == "sampler2D" || tkn1 == "sampler3D");


                var arg_name;
                var is_array = false;
                var k1 = tkn2.indexOf("[");
                if (k1 != -1) {
                    arg_name = tkn2.substring(0, k1);
                    is_array = true;
                }
                else{
                    var k2 = tkn2.indexOf(";");
                    if (k2 != -1) {
                        arg_name = tkn2.substring(0, k2);
                    }
                    else{
                        arg_name = tkn2;
                    }
                }

                var arg_val = param.args[arg_name];
                Assert(arg_val != undefined);

                var arg_inf = { name: arg_name, value: arg_val, type: tkn1, isArray: is_array };

                switch (tokens[0]) {
                    case "in":
                        pkg.attributes.push(arg_inf);
                        break;

                    case "uniform":
                        if (tkn1 == "sampler2D" || tkn1 == "sampler3D") {

                            pkg.textures.push(arg_inf);
                        }
                        else {
                            pkg.uniforms.push(arg_inf);

                        }
                        break;

                    case "out":
                        pkg.varyings.push(arg_inf);
                        break;
                }
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

        makeShader(type, source) {
            var shader = gl.createShader(type); chk();
            gl.shaderSource(shader, source); chk();
            gl.compileShader(shader); chk();

            return shader;
        }

        makeTexture(tex_id, texture_inf) {
            var dim = texture_inf.type == "sampler3D" ? gl.TEXTURE_3D : gl.TEXTURE_2D;

            var texture = gl.createTexture(); chk();

            gl.activeTexture(tex_id); chk();
            gl.bindTexture(dim, texture); chk();

            gl.texParameteri(dim, gl.TEXTURE_MAG_FILTER, gl.NEAREST); chk();
            gl.texParameteri(dim, gl.TEXTURE_MIN_FILTER, gl.NEAREST); chk();
            gl.texParameteri(dim, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); chk();
            gl.texParameteri(dim, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE); chk();

            return texture;
        }

        setTextureData(texture_inf, tex_id, texture) {
            var dim = texture_inf.type == "sampler3D" ? gl.TEXTURE_3D : gl.TEXTURE_2D;

            gl.activeTexture(tex_id); chk();
            gl.bindTexture(dim, texture); chk();
            if (dim == gl.TEXTURE_2D) {

                gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, texture_inf.value.Cols / 4, texture_inf.value.Rows, 0, gl.RGBA, gl.FLOAT, texture_inf.value.dt); chk();
            }
            else {
                Assert(dim == gl.TEXTURE_3D, "Set-Tex");

                gl.texImage3D(gl.TEXTURE_3D, 0, gl.RGBA32F, texture_inf.value.Cols / 4, texture_inf.value.Rows, texture_inf.value.Depth, 0, gl.RGBA, gl.FLOAT, texture_inf.value.dt); chk();
            }
        }

        vecDim(tp) {
            if (tp == "vec4") {
                return 4;
            }
            else if (tp == "vec3") {
                return 3;
            }
            else if (tp == "vec2") {
                return 2;
            }
            else {
                return 1;
            }
        }

        makePackage(param) {
            var pkg = {};
            this.packages[param.key] = pkg;

            pkg.key = param.key;

            this.parseShader(pkg, param);

            var fsrc = Shaders['fs-transform'];
            var vertex_shader = this.makeShader(gl.VERTEX_SHADER, param.shaderText);

            var fragment_shader = this.makeShader(gl.FRAGMENT_SHADER, fsrc);

            pkg.program = this.makeProgram(vertex_shader, fragment_shader, pkg.varyings);
            gl.useProgram(pkg.program); chk();

            // ユニフォーム変数の初期処理
            pkg.locUniforms = [];
            for(let u of pkg.uniforms) {

                var loc = gl.getUniformLocation(pkg.program, u.name); chk();
                pkg.locUniforms.push(loc);
            }

            // テクスチャの初期処理
            pkg.locTextures = [];
            pkg.Textures = [];
            for (var i = 0; i < pkg.textures.length; i++) {

                var loc = gl.getUniformLocation(pkg.program, pkg.textures[i].name); chk();
                pkg.locTextures.push(loc);

                var tex = this.makeTexture(this.TEXTUREs[i], pkg.textures[i]);
                pkg.Textures.push(tex);
            }

            pkg.attribElementCount = param.elementCount;

            pkg.AttribBuffers = [];
            for (let attrib of pkg.attributes) {
                var attrib_dim = this.vecDim(attrib.type);
                var attrib_len = attrib.value instanceof Mat ? attrib.value.dt.length : attrib.value.length;
                var elemen_count = attrib_len / attrib_dim;

                if (pkg.elementCount == undefined) {
                    pkg.attribElementCount = elemen_count;
                }
                else {

                    Assert(pkg.elementCount == elemen_count);
                }

                var vbo = gl.createBuffer();
                pkg.AttribBuffers.push(vbo);
            }

            pkg.feedbackBuffers = [];

            for (let varying of pkg.varyings) {
                var out_buffer_size = this.vecDim(varying.type) * pkg.attribElementCount * Float32Array.BYTES_PER_ELEMENT;

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

        setUniformsData(pkg) {
            for (var i = 0; i < pkg.uniforms.length; i++) {
                var u = pkg.uniforms[i];
                if (u.value instanceof Mat || u.value instanceof Float32Array) {

                    var val = u.value instanceof Mat ? u.value.dt : u.value;

                    switch (this.vecDim(u.type)) {
                        case 4:
                            if (u.isArray) {

                                gl.uniform4fv(pkg.locUniforms[i], val); chk();
                            }
                            else {

                                gl.uniform4f(pkg.locUniforms[i], val[0], val[1], val[2], val[3]); chk();
                            }
                            break;

                        case 3:
                            if (u.isArray) {

                                gl.uniform3fv(pkg.locUniforms[i], val); chk();
                            }
                            else {

                                gl.uniform3f(pkg.locUniforms[i], val[0], val[1], val[2]); chk();
                            }
                            break;
                        case 2:
                            if (u.isArray) {

                                gl.uniform2fv(pkg.locUniforms[i], val); chk();
                            }
                            else {

                                gl.uniform2f(pkg.locUniforms[i], val[0], val[1]); chk();
                            }
                            break;
                        case 1:
                            if (u.isArray) {

                                gl.uniform1fv(pkg.locUniforms[i], val); chk();
                            }
                            else {

                                gl.uniform1f(pkg.locUniforms[i], val[0]); chk();
                            }
                            break;
                    }
                }
                else {

                    if (u.type == "int") {

                        gl.uniform1i(pkg.locUniforms[i], u.value); chk();
                    }
                    else {

                        gl.uniform1f(pkg.locUniforms[i], u.value); chk();
                    }
                }
            }
        }

        compute(param) {
            var pkg = this.packages[param.key];
            if (!pkg) {

                pkg = this.makePackage(param);
            }
            else {

                gl.useProgram(pkg.program); chk();
            }

            for(let args of[ pkg.attributes, pkg.uniforms, pkg.textures, pkg.varyings ]) {
                for (let arg of args) {
                    var val = param.args[arg.name];
                    Assert(val != undefined);
                    arg.value = val;
                }
            }

            // -- Init Buffer
            for (var i = 0; i < pkg.attributes.length; i++) { 
                var attrib = pkg.attributes[i];
                var dim = this.vecDim(attrib.type);

                gl.bindBuffer(gl.ARRAY_BUFFER, pkg.AttribBuffers[i]); chk();
                gl.vertexAttribPointer(i, dim, gl.FLOAT, false, 4 * dim, 0); chk();
                gl.enableVertexAttribArray(i); chk();
                gl.bindAttribLocation(pkg.program, i, attrib.name);
                gl.bufferData(gl.ARRAY_BUFFER, attrib.value, gl.STATIC_DRAW);
            }

            gl.useProgram(pkg.program); chk();

            gl.enable(gl.RASTERIZER_DISCARD); chk();

            gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, pkg.transformFeedback); chk();

            // テクスチャの値のセット
            for (var i = 0; i < pkg.textures.length; i++) {

                this.setTextureData(pkg.textures[i], this.TEXTUREs[i], pkg.Textures[i]);
                gl.uniform1i(pkg.locTextures[i], i); chk();
            }

            // ユニフォーム変数のセット
            this.setUniformsData(pkg);

            for (var i = 0; i < pkg.varyings.length; i++) {

                gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, i, pkg.feedbackBuffers[i]); chk();
            }

            // 計算開始
            gl.beginTransformFeedback(gl.POINTS); chk();    // TRIANGLES
            gl.drawArrays(gl.POINTS, 0, pkg.attribElementCount); chk();
            gl.endTransformFeedback(); chk();

            gl.disable(gl.RASTERIZER_DISCARD); chk();

            for (var i = 0; i < pkg.varyings.length; i++) {

                gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, i, null); chk();

                // 処理結果を表示
                gl.bindBuffer(gl.ARRAY_BUFFER, pkg.feedbackBuffers[i]); chk();

                var out_buf = pkg.varyings[i].value;
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
