// JavaScript source code

function MakeFloat32Index(n) {
    var v = new Float32Array(n);
    for (var i = 0; i < n; i++) {
        v[i] = i;
    }

    return v;
}

function CreateWebGLLib(canvas) {
    let gl;

    function chk() {
        Assert(gl.getError() == gl.NO_ERROR);
    }

    class WebGLLib {

        constructor(canvas) {
            console.log("init WebGL");

            this.packages = {};

            if (!canvas) {

                // -- Init Canvas
                canvas = document.createElement('canvas');
                canvas.width = 32;
                canvas.height = 32;
                document.body.appendChild(canvas);
            }

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

        getGL() {
            return gl;
        }

        WebGLClear() {
            for (var key in this.packages) {
                var pkg = this.packages[key];

                gl.bindBuffer(gl.ARRAY_BUFFER, null); chk();
                gl.deleteBuffer(pkg.idxBuffer); chk();

                for (let varying of pkg.varyings) {
                    if (varying.feedbackBuffer) {
                        gl.deleteBuffer(varying.feedbackBuffer); chk();
                    }
                }

                gl.deleteTransformFeedback(pkg.transformFeedback); chk();

                gl.bindTexture(gl.TEXTURE_2D, null); chk();
                gl.bindTexture(gl.TEXTURE_3D, null); chk();

                pkg.textures.forEach(x => gl.deleteTexture(x.Texture), chk())

                gl.deleteProgram(pkg.program); chk();
                console.log("clear pkg:" + pkg.key);
            }
        }

        parseShader(pkg, param) {
            pkg.attributes = [];
            pkg.uniforms = [];
            pkg.textures = [];
            pkg.varyings = [];
            var lines = param.vertexShader.split(/(\r\n|\r|\n)+/);
            for(let line of lines) {

                var tokens = line.trim().split(/[\s\t]+/);
                if (tokens.length < 3) {
                    continue;
                }

                var tkn0 = tokens[0];
                var tkn1 = tokens[1];
                var tkn2 = tokens[2];

                if (tkn0 != "in" && tkn0 != "uniform" && tkn0 != "out") {
                    continue;
                }
                Assert(tkn1 == "int" || tkn1 == "float" || tkn1 == "vec2" || tkn1 == "vec3" || tkn1 == "vec4" ||
                    tkn1 == "sampler2D" || tkn1 == "sampler3D" ||
                    tkn1 == "mat4" || tkn1 == "mat3" || tkn1 == "bool");


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
                if(arg_val == undefined){
                    if(tokens[0] == "out"){
                        continue;
                    }
                }

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

            if (varyings) {

                var varying_names = varyings.map(x => x.name);
                gl.transformFeedbackVaryings(prg, varying_names, gl.SEPARATE_ATTRIBS); chk();   // gl.INTERLEAVED_ATTRIBS 
            }

            gl.linkProgram(prg); chk();


            if (!gl.getProgramParameter(prg, gl.LINK_STATUS)) {
                console.log("Link Error:" + gl.getProgramInfoLog(prg));
            }


            gl.deleteShader(vshaderTransform); chk();
            gl.deleteShader(fshaderTransform); chk();

            return prg;
        }

        makeShader(type, source) {
            var shader = gl.createShader(type); chk();
            gl.shaderSource(shader, source); chk();
            gl.compileShader(shader); chk();

            if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
                alert(gl.getShaderInfoLog(shader));
                return null;
            }

            return shader;
        }

        makeAttrib(pkg) {
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

                attrib.AttribBuffer = gl.createBuffer();
                attrib.AttribLoc = gl.getAttribLocation(pkg.program, attrib.name); chk();
                gl.enableVertexAttribArray(attrib.AttribLoc); chk();
                gl.bindAttribLocation(pkg.program, attrib.AttribLoc, attrib.name);
            }
        }

        makeTexture(pkg) {
            for (var i = 0; i < pkg.textures.length; i++) {
                var tex_inf = pkg.textures[i];

                tex_inf.locTexture = gl.getUniformLocation(pkg.program, tex_inf.name); chk();

                var dim = tex_inf.type == "sampler3D" ? gl.TEXTURE_3D : gl.TEXTURE_2D;

                tex_inf.Texture = gl.createTexture(); chk();

                gl.activeTexture(this.TEXTUREs[i]); chk();
                gl.bindTexture(dim, tex_inf.Texture); chk();

                gl.texParameteri(dim, gl.TEXTURE_MAG_FILTER, gl.NEAREST); chk();
                gl.texParameteri(dim, gl.TEXTURE_MIN_FILTER, gl.NEAREST); chk();
                gl.texParameteri(dim, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); chk();
                gl.texParameteri(dim, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE); chk();
            }
        }

        setTextureData(pkg) {
            for (var i = 0; i < pkg.textures.length; i++) {
                var tex_inf = pkg.textures[i];

                gl.uniform1i(tex_inf.locTexture, i); chk();

                var dim = tex_inf.type == "sampler3D" ? gl.TEXTURE_3D : gl.TEXTURE_2D;

                gl.activeTexture(this.TEXTUREs[i]); chk();
                gl.bindTexture(dim, tex_inf.Texture); chk();
                if (dim == gl.TEXTURE_2D) {

                    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, tex_inf.value.Cols / 4, tex_inf.value.Rows, 0, gl.RGBA, gl.FLOAT, tex_inf.value.dt); chk();
                }
                else {
                    Assert(dim == gl.TEXTURE_3D, "Set-Tex");

                    gl.texImage3D(gl.TEXTURE_3D, 0, gl.RGBA32F, tex_inf.value.Cols / 4, tex_inf.value.Rows, tex_inf.value.Depth, 0, gl.RGBA, gl.FLOAT, tex_inf.value.dt); chk();
                }
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

        initUniform(pkg) {
            pkg.uniforms.forEach(u => u.locUniform = gl.getUniformLocation(pkg.program, u.name), chk());
        }

        makePackage(param) {
            var pkg = {};
            this.packages[param.key] = pkg;

            pkg.key = param.key;

            this.parseShader(pkg, param);

            var fsrc = Shaders['fs-transform'];
            var vertex_shader = this.makeShader(gl.VERTEX_SHADER, param.vertexShader);

            var fragment_shader = this.makeShader(gl.FRAGMENT_SHADER, fsrc);

            pkg.program = this.makeProgram(vertex_shader, fragment_shader, pkg.varyings);
            gl.useProgram(pkg.program); chk();

            // ユニフォーム変数の初期処理
            this.initUniform(pkg);

            // テクスチャの初期処理
            this.makeTexture(pkg);

            pkg.attribElementCount = param.elementCount;

            this.makeAttrib(pkg);

            for (let varying of pkg.varyings) {
                var out_buffer_size = this.vecDim(varying.type) * pkg.attribElementCount * Float32Array.BYTES_PER_ELEMENT;

                // Feedback empty buffer
                varying.feedbackBuffer = gl.createBuffer(); chk();
                gl.bindBuffer(gl.ARRAY_BUFFER, varying.feedbackBuffer); chk();
                gl.bufferData(gl.ARRAY_BUFFER, out_buffer_size, gl.STATIC_COPY); chk();
                gl.bindBuffer(gl.ARRAY_BUFFER, null); chk();
            }

            // -- Init TransformFeedback 
            pkg.transformFeedback = gl.createTransformFeedback(); chk();

            return pkg;
        }

        setAttribData(pkg) {
            // -- Init Buffer
            for (var i = 0; i < pkg.attributes.length; i++) {
                var attrib = pkg.attributes[i];
                var dim = this.vecDim(attrib.type);

                gl.bindBuffer(gl.ARRAY_BUFFER, attrib.AttribBuffer); chk();
                gl.vertexAttribPointer(attrib.AttribLoc, dim, gl.FLOAT, false, 0, 0); chk();
                gl.bufferData(gl.ARRAY_BUFFER, attrib.value, gl.STATIC_DRAW);
            }
        }

        setUniformsData(pkg) {
            for (var i = 0; i < pkg.uniforms.length; i++) {
                var u = pkg.uniforms[i];
                if (u.value instanceof Mat || u.value instanceof Float32Array) {

                    var val = u.value instanceof Mat ? u.value.dt : u.value;

                    switch (u.type) {
                        case "mat4":
                            gl.uniformMatrix4fv(u.locUniform, false, val); chk();
                            break;
                        case "mat3":
                            gl.uniformMatrix3fv(u.locUniform, false, val); chk();
                            break;
                        case "vec4":
                            gl.uniform4fv(u.locUniform, val); chk();
                            break;
                        case "vec3":
                            gl.uniform3fv(u.locUniform, val); chk();
                            break;
                        case "vec2":
                            gl.uniform2fv(u.locUniform, val); chk();
                            break;
                        case "float":
                            gl.uniform1fv(u.locUniform, val); chk();
                            break;
                        default:
                            Assert(false);
                            break;
                    }
                }
                else {

                    if (u.type == "int" || u.type == "bool") {

                        gl.uniform1i(u.locUniform, u.value); chk();
                    }
                    else {

                        gl.uniform1f(u.locUniform, u.value); chk();
                    }
                }
            }
        }

        copyParamArgsValue(param, pkg){
            for(let args of[ pkg.attributes, pkg.uniforms, pkg.textures, pkg.varyings ]) {
                for (let arg of args) {
                    var val = param.args[arg.name];
                    Assert(val != undefined);
                    arg.value = val;
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

            this.copyParamArgsValue(param, pkg);

            this.setAttribData(pkg);

            gl.useProgram(pkg.program); chk();

            gl.enable(gl.RASTERIZER_DISCARD); chk();

            gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, pkg.transformFeedback); chk();

            // テクスチャの値のセット
            this.setTextureData(pkg);

            // ユニフォーム変数のセット
            this.setUniformsData(pkg);

            for (var i = 0; i < pkg.varyings.length; i++) {
                var varying = pkg.varyings[i];
                gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, i, varying.feedbackBuffer); chk();
            }

            // 計算開始
            gl.beginTransformFeedback(gl.POINTS); chk();    // TRIANGLES
            gl.drawArrays(gl.POINTS, 0, pkg.attribElementCount); chk();
            gl.endTransformFeedback(); chk();

            gl.disable(gl.RASTERIZER_DISCARD); chk();

            for (var i = 0; i < pkg.varyings.length; i++) {
                varying = pkg.varyings[i];

                gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, i, null); chk();

                // 処理結果を表示
                gl.bindBuffer(gl.ARRAY_BUFFER, varying.feedbackBuffer); chk();

                var out_buf = varying.value;
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

    return new WebGLLib(canvas);
}
