// JavaScript source code

function Assert(b, msg) {
    if (!b) {
        console.log(msg);
    }
};

function MakeFloat32Index(n) {
    var v = new Float32Array(n);
    for (var i = 0; i < n; i++) {
        v[i] = i;
    }

    return v;
}

function make2DArray(nrow, ncol, init) {
    var v;
    
    if(init){
        if (init instanceof Float32Array) {

            v = init;
        }
        else {

            v = new Float32Array(init);
        }

        Assert(v.length == nrow * ncol);
    }
    else{

        v = new Float32Array(nrow * ncol);
    }

    v.nrow  = nrow;
    v.ncol  = ncol;

    v.shape = [nrow, ncol];

    v.T = function () {
        var m = make2DArray(this.ncol, this.nrow);
        var i1 = 0;
        for(var r = 0; r < this.ncol; r++) {
            var i2 = r;
            for (var c = 0; c < this.nrow; c++) {
                m[i1] = this[i2];
                i1++;
                i2 += this.ncol;
            }
        }

        return m;
    }

    return v;
}

class TextureInfo {
    constructor(texel_type, value) {
        this.texelType = texel_type;
        this.value     = value;
    }
}

class ArrayView {
    constructor() {
        var args;

        if(arguments.length == 1 && Array.isArray(arguments[0])) {

            args = arguments[0];
        }
        else {

            // 引数のリストをArrayに変換します。
            args = Array.prototype.slice.call(arguments);
        }

        // 引数の最後
        var last_arg = args[args.length -1];
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

        this.ncol = this.shape[this.shape.length -1];
        if (this.shape.length == 1) {

            this.nrow = 1;
        }
        else {

            this.nrow = this.shape[this.shape.length -2];
        }

        if (!this.dt) {
            this.dt = new Float32Array(this.shape.reduce((x, y) => x * y));
        }
    }

    Map(f) {
        return new ArrayView(this.nrow, this.ncol, this.dt.map(f));
    }

    T() {
        var m = new ArrayView(this.ncol, this.nrow);
        var i1 = 0;
        for (var r = 0; r < this.ncol; r++) {
            var i2 = r;
            for (var c = 0; c < this.nrow; c++) {
                m.dt[i1] = this.dt[i2];
                i1++;
                i2 += this.ncol;
            }
        }

        return m;
    }

    At2(r, c) {
        Assert(r < this.nrow && c < this.ncol, "ArrayView-at");
        return this.dt[r * this.ncol +c];
    }

    Set2(r, c, val) {
        Assert(r < this.nrow && c < this.ncol, "ArrayView-set");

        this.dt[r * this.ncol +c]= val;
    }

    At3(d, r, c) {
        Assert(d < this.shape[this.shape.length -3]&& r < this.nrow && c < this.ncol, "ArrayView-at3");

        return this.dt[(d * this.nrow +r) * this.ncol +c];
    }

    Set3(d, r, c, val) {
        Assert(d < this.shape[this.shape.length -3]&& r < this.nrow && c < this.ncol, "ArrayView-set3");

        this.dt[(d * this.nrow +r) * this.ncol +c]= val;
    }

    Col(c) {
        var v = new Float32Array(this.nrow);
        for (var r = 0; r < this.nrow; r++) {
            v[r]= this.dt[r * this.ncol +c];
        }

        return new ArrayView(this.nrow, 1, v);
    }

    Add(m) {
        Assert(m instanceof ArrayView && m.nrow == this.nrow && m.ncol == this.ncol, "ArrayView-add");
        var v = new Float32Array(this.nrow * this.ncol);
        for (var r = 0; r < this.nrow; r++) {
            for (var c = 0; c < this.ncol; c++) {
                var k = r * this.ncol +c;
                v[k] = this.dt[k]+m.dt[k];
            }
        }

        return new ArrayView(this.nrow, this.ncol, v);
    }

    AddVec(vec) {
        Assert(vec instanceof ArrayView && vec.nrow == this.nrow && vec.ncol == 1, "ArrayView-add-V");
        var v = new Float32Array(this.nrow * this.ncol);
        for (var r = 0; r < this.nrow; r++) {
            for (var c = 0; c < this.ncol; c++) {
                var k = r * this.ncol + c;
                v[k] = this.dt[k] + vec.dt[r];
            }
        }

        return new ArrayView(this.nrow, this.ncol, v);
    }

    Reduce(f) {
        var v = new Float32Array(this.nrow);
        for (var r = 0; r < this.nrow; r++) {
            var x;
            for (var c = 0; c < this.ncol; c++) {
                var k = r * this.ncol +c;
                if (c == 0) {

                    x = this.dt[k];
                }
                else {

                    x = f(x, this.dt[k]);
                }
            }
            v[r]= x;
        }

        return new ArrayView(this.nrow, 1, v);
    }

    Sub(m) {
        Assert(m instanceof ArrayView && m.nrow == this.nrow && m.ncol == this.ncol, "ArrayView-Sub");
        var v = new Float32Array(this.nrow * this.ncol);
        for (var r = 0; r < this.nrow; r++) {
            for (var c = 0; c < this.ncol; c++) {
                var k = r * this.ncol + c;
                v[k]= this.dt[k]-m.dt[k];
            }
        }

        return new ArrayView(this.nrow, this.ncol, v);
    }

    Mul(m) {
        if (m instanceof Number) {

            return new ArrayView(this.nrow, this.ncol, this.dt.map(x => x * m));
            }
        Assert(m instanceof ArrayView && m.nrow == this.nrow && m.ncol == this.ncol && m.columnMajor == this.columnMajor, "Array-View-mul");
        var v = new Float32Array(this.nrow * this.ncol);
        for (var r = 0; r < this.nrow; r++) {
            for(var c = 0; c < this.ncol; c++) {
                var k = r * this.ncol +c;
                v[k]= this.dt[k]* m.dt[k];
                }
        }

        return new ArrayView(this.nrow, this.ncol, v);
    }

    Dot(m) {
        Assert(m instanceof ArrayView && m.nrow == this.ncol, "ArrayView-Dot");

        var v = new Float32Array(this.nrow * m.ncol);
        for (var r = 0; r < this.nrow; r++) {
            for(var c = 0; c < m.ncol; c++) {
                var sum = 0;
                for (var k = 0; k < this.ncol; k++) {
                    sum += this.dt[r * this.ncol +k]* m.dt[k * m.ncol +c];
                }
                v[r * m.ncol +c]= sum;
                }
                }
        return new ArrayView(this.nrow, m.ncol, v);
    }
}

function CreateGPGPU(canvas) {
    let gl;

    function chk() {
        Assert(gl.getError() == gl.NO_ERROR);
    }

    class GPGPU {
        constructor(canvas) {
            console.log("init WebGL");
            this.setStandardShader();

            this.packages = {};

            if (!canvas) {

                // -- Init Canvas
                canvas = document.createElement('canvas');
                canvas.width = 32;
                canvas.height = 32;
                document.body.appendChild(canvas);
            }
            this.canvas = canvas;

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
            for (var id in this.packages) {
                var pkg = this.packages[id];

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
                console.log("clear pkg:" + pkg.id);
            }
        }

        parseShader(pkg, param) {
            // attribute変数、uniform変数、テクスチャ、varying変数の配列を初期化する。
            pkg.attributes = [];
            pkg.uniforms = [];
            pkg.textures = [];
            pkg.varyings = [];

            // 頂点シェーダとフラグメントシェーダのソースに対し
            for(let shader_text of[ param.vertexShader,  param.fragmentShader ]) {

                // 行ごとに分割する。
                var lines = shader_text.split(/(\r\n|\r|\n)+/);

                // すべての行に対し
                for(let line of lines) {

                    // 行を空白で分割する。
                    var tokens = line.trim().split(/[\s\t]+/);

                    if (tokens.length < 3) {
                        // トークンの長さが3未満の場合
                        continue;
                    }

                    // 最初、2番目、3番目のトークン
                    var tkn0 = tokens[0];
                    var tkn1 = tokens[1];
                    var tkn2 = tokens[2];

                    if (tkn0 != "in" && tkn0 != "uniform" && tkn0 != "out") {
                        // 最初のトークンが in, uniform, out でない場合
                        continue;
                    }

                    if (shader_text == param.fragmentShader && tkn0 != "uniform") {
                        // フラグメントシェーダで uniform でない場合 ( フラグメントシェーダの入力(in)と出力(out)はアプリ側では使わない。 )

                        continue;
                    }
                    Assert(tkn1 == "int" || tkn1 == "float" || tkn1 == "vec2" || tkn1 == "vec3" || tkn1 == "vec4" ||
                        tkn1 == "sampler2D" || tkn1 == "sampler3D" ||
                        tkn1 == "mat4" || tkn1 == "mat3" || tkn1 == "bool");


                    var arg_name;
                    var is_array = false;
                    var k1 = tkn2.indexOf("[");
                    if (k1 != -1) {
                        // 3番目のトークンが [ を含む場合

                        // 配列と見なす。
                        is_array = true;

                        // 変数名を得る。
                        arg_name = tkn2.substring(0, k1);
                    }
                    else{
                        // 3番目のトークンが [ を含まない場合

                        var k2 = tkn2.indexOf(";");
                        if (k2 != -1) {
                            // 3番目のトークンが ; を含む場合

                            // 変数名を得る。
                            arg_name = tkn2.substring(0, k2);
                        }
                        else{
                            // 3番目のトークンが ; を含まない場合

                            // 変数名を得る。
                            arg_name = tkn2;
                        }
                    }

                    // 変数の値を得る。
                    var arg_val = param.args[arg_name];

                    if (arg_val == undefined) {
                        if(tokens[0] == "out"){
                            continue;
                        }
                    }

                    if (tkn1 == "sampler2D" || tkn1 == "sampler3D") {
                        // テクスチャのsamplerの場合

                        Assert(tokens[0] == "uniform" && arg_val instanceof TextureInfo);

                        // 変数名をセットする。
                        arg_val.name = arg_name;

                        // samplerのタイプをセットする。
                        arg_val.samplerType = tkn1;

                        // 配列かどうかをセットする。
                        arg_val.isArray = is_array;

                        // テクスチャの配列に追加する。
                        pkg.textures.push(arg_val);
                    }
                    else {
                        // テクスチャのsamplerでない場合

                        // 変数の名前、値、型、配列かどうかをセットする。
                        var arg_inf = { name: arg_name, value: arg_val, type: tkn1, isArray: is_array };

                        switch (tokens[0]) {
                            case "in":
                                // attribute変数の場合

                                pkg.attributes.push(arg_inf);
                                break;

                            case "uniform":
                                // uniform変数の場合

                                pkg.uniforms.push(arg_inf);
                                break;

                            case "out":
                                // varying変数の場合

                                pkg.varyings.push(arg_inf);
                                break;
                        }
                    }
                }
            }
        }

        /*
            プログラムを作る。
        */
        makeProgram(vertex_shader, fragment_shader, varyings) {
            // プログラムを作る。
            var prg = gl.createProgram(); chk();

            // 頂点シェーダをアタッチする。
            gl.attachShader(prg, vertex_shader); chk();

            // フラグメントシェーダをアタッチする。
            gl.attachShader(prg, fragment_shader); chk();

            if (varyings) {
                // varying変数がある場合

                // varying変数の名前の配列
                var varying_names = varyings.map(x => x.name);

                // Transform Feedbackで使うvarying変数を指定する。
                gl.transformFeedbackVaryings(prg, varying_names, gl.SEPARATE_ATTRIBS); chk();   // gl.INTERLEAVED_ATTRIBS 
            }

            // プログラムをリンクする。
            gl.linkProgram(prg); chk();

            if (!gl.getProgramParameter(prg, gl.LINK_STATUS)) {
                // リンクエラーがある場合

                console.log("Link Error:" + gl.getProgramInfoLog(prg));
            }

            // 頂点シェーダを削除する。
            gl.deleteShader(vertex_shader); chk();

            // フラグメントシェーダを削除する。
            gl.deleteShader(fragment_shader); chk();

            return prg;
        }

        /*        
            シェーダを作る。
        */
        makeShader(type, source) {
            source = "#version 300 es\nprecision highp float;\nprecision highp int;\n" + source;

            // シェーダを作る。
            var shader = gl.createShader(type); chk();

            // シェーダにソースをセットする。
            gl.shaderSource(shader, source); chk();

            // シェーダをコンパイルする。
            gl.compileShader(shader); chk();

            if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
                // コンパイル エラーの場合

                alert(gl.getShaderInfoLog(shader));
                return null;
            }

            return shader;
        }

        /*
            attribute変数を作る。
        */
        makeAttrib(pkg) {
            // すべてのattribute変数に対し
            for (let attrib of pkg.attributes) {
                // attribute変数の次元
                var attrib_dim = this.vecDim(attrib.type);

                // attribute変数の配列の長さ
                var attrib_len = attrib.value instanceof ArrayView ? attrib.value.dt.length : attrib.value.length;

                // 要素の個数
                var elemen_count = attrib_len / attrib_dim;

                if (pkg.elementCount == undefined) {
                    pkg.attribElementCount = elemen_count;
                }
                else {

                    Assert(pkg.elementCount == elemen_count);
                }

                // バッファを作る。
                attrib.AttribBuffer = gl.createBuffer();

                // attribute変数の位置
                attrib.AttribLoc = gl.getAttribLocation(pkg.program, attrib.name); chk();

                // 指定した位置のattribute配列を有効にする。
                gl.enableVertexAttribArray(attrib.AttribLoc); chk();

                // attribute変数の位置と変数名をバインドする。
                gl.bindAttribLocation(pkg.program, attrib.AttribLoc, attrib.name);
            }
        }

        /*
            テクスチャを作る。
        */
        makeTexture(pkg) {
            // すべてのテクスチャに対し
            for (var i = 0; i < pkg.textures.length; i++) {
                var tex_inf = pkg.textures[i];

                // テクスチャのuniform変数の位置
                tex_inf.locTexture = gl.getUniformLocation(pkg.program, tex_inf.name); chk();

                var dim = tex_inf.samplerType == "sampler3D" ? gl.TEXTURE_3D : gl.TEXTURE_2D;

                // テクスチャを作る。
                tex_inf.Texture = gl.createTexture(); chk();

                // 指定した位置のテクスチャをアクティブにする。
                gl.activeTexture(this.TEXTUREs[i]); chk();

                // 作成したテクスチャをバインドする。
                gl.bindTexture(dim, tex_inf.Texture); chk();

                if (tex_inf.value instanceof Image) {
                    // テクスチャが画像の場合

                    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR); chk();
                    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_NEAREST); chk();

                    //        gl.texParameteri(gl.TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_MIRRORED_REPEAT); //GL_REPEAT
                    //        gl.texParameteri(gl.TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_MIRRORED_REPEAT); //GL_REPEAT

                    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true); chk();
                    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, tex_inf.value); chk();
                    gl.generateMipmap(gl.TEXTURE_2D); chk();
                }
                else {
                    // テクスチャが画像でない場合

                    gl.texParameteri(dim, gl.TEXTURE_MAG_FILTER, gl.NEAREST); chk();
                    gl.texParameteri(dim, gl.TEXTURE_MIN_FILTER, gl.NEAREST); chk();
                    gl.texParameteri(dim, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); chk();
                    gl.texParameteri(dim, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE); chk();
                }
            }
        }

        /*
            テクスチャのデータをセットする。
        */
        setTextureData(pkg) {
            for (var i = 0; i < pkg.textures.length; i++) {
                var tex_inf = pkg.textures[i];

                // テクスチャのuniform変数にテクスチャの番号をセットする。
                gl.uniform1i(tex_inf.locTexture, i); chk();

                var dim = tex_inf.samplerType == "sampler3D" ? gl.TEXTURE_3D : gl.TEXTURE_2D;

                // 指定した位置のテクスチャをアクティブにする。
                gl.activeTexture(this.TEXTUREs[i]); chk();

                // テクスチャをバインドする。
                gl.bindTexture(dim, tex_inf.Texture); chk();

                if (tex_inf.value instanceof Image) {
                    // テクスチャが画像の場合

                }
                else {
                    // テクスチャが画像でない場合

                    var data;
                    if (tex_inf.value instanceof ArrayView) {
                        data = tex_inf.value.dt;
                    }
                    else {
                        data = tex_inf.value;
                    }

                    var internal_format, format, col_size;
                    switch (tex_inf.texelType) {
                        case "float":
                            internal_format = gl.R32F;
                            format = gl.RED;
                            col_size = 1;
                            break;

                        case "vec2":
                            internal_format = gl.RG32F;
                            format = gl.RG;
                            col_size = 2;
                            break;

                        case "vec3":
                            internal_format = gl.RGB32F;
                            format = gl.RGB;
                            col_size = 3;
                            break;

                        case "vec4":
                            internal_format = gl.RGBA32F;
                            format = gl.RGBA;
                            col_size = 4;
                            break;

                        default:
                            Assert(false);
                            break;
                    }

                    if (dim == gl.TEXTURE_2D) {

                        gl.texImage2D(gl.TEXTURE_2D, 0, internal_format, tex_inf.value.ncol / col_size, tex_inf.value.nrow, 0, format, gl.FLOAT, data); chk();
                    }
                    else {
                        Assert(dim == gl.TEXTURE_3D, "set-Tex");

                        gl.texImage3D(gl.TEXTURE_3D, 0, internal_format, tex_inf.value.ncol / col_size, tex_inf.value.nrow, tex_inf.value.shape[tex_inf.value.shape.length - 3], 0, format, gl.FLOAT, data); chk();
                    }
                }
            }
        }

        makeVertexIndexBuffer(pkg, param) {
            gl.clearColor(0.0, 0.0, 0.0, 1.0); chk();
            gl.enable(gl.DEPTH_TEST); chk();

            var buf = gl.createBuffer(); chk();
            gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buf); chk();
            gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, param.VertexIndexBuffer, gl.STATIC_DRAW); chk();

            pkg.VertexIndexBufferInf = {
                value: param.VertexIndexBuffer,
                buffer: buf
            };
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

        build(param) {
            var pkg = {};
            this.packages[param.id] = pkg;

            pkg.id = param.id;

            if (!param.fragmentShader) {
                // フラグメントシェーダが指定されてない場合

                // デフォルトのフラグメントシェーダをセットする。
                param.fragmentShader = this.minFragmentShader;
            }

            this.parseShader(pkg, param);

            // 頂点シェーダを作る。
            var vertex_shader = this.makeShader(gl.VERTEX_SHADER, param.vertexShader);

            // フラグメントシェーダを作る。
            var fragment_shader = this.makeShader(gl.FRAGMENT_SHADER, param.fragmentShader);

            // プログラムを作る。
            pkg.program = this.makeProgram(vertex_shader, fragment_shader, pkg.varyings);

            // プログラムを使用する。
            gl.useProgram(pkg.program); chk();

            // ユニフォーム変数の初期処理
            this.initUniform(pkg);

            // テクスチャを作る。
            this.makeTexture(pkg);

            pkg.attribElementCount = param.elementCount;

            // attribute変数を作る。
            this.makeAttrib(pkg);

            if (pkg.varyings.length != 0) {
                //  varying変数がある場合

                // すべてのvarying変数に対し
                for (let varying of pkg.varyings) {
                    var out_buffer_size = this.vecDim(varying.type) * pkg.attribElementCount * Float32Array.BYTES_PER_ELEMENT;

                    // Transform Feedbackバッファを作る。
                    varying.feedbackBuffer = gl.createBuffer(); chk();

                    // バッファをバインドする。
                    gl.bindBuffer(gl.ARRAY_BUFFER, varying.feedbackBuffer); chk();
                    gl.bufferData(gl.ARRAY_BUFFER, out_buffer_size, gl.STATIC_COPY); chk();
                    gl.bindBuffer(gl.ARRAY_BUFFER, null); chk();
                }

                // Transform Feedbackを作る。
                pkg.transformFeedback = gl.createTransformFeedback(); chk();
            }

            if (param.VertexIndexBuffer) {
                this.makeVertexIndexBuffer(pkg, param);
            }

            return pkg;
        }

        setAttribData(pkg) {
            // -- Init Buffer
            for (let attrib of pkg.attributes) {
                var dim = this.vecDim(attrib.type);

                gl.bindBuffer(gl.ARRAY_BUFFER, attrib.AttribBuffer); chk();
                gl.vertexAttribPointer(attrib.AttribLoc, dim, gl.FLOAT, false, 0, 0); chk();
                gl.bufferData(gl.ARRAY_BUFFER, attrib.value, gl.STATIC_DRAW);
            }
        }

        setUniformsData(pkg) {
            for (let u of pkg.uniforms) {
                if (u.value instanceof ArrayView || u.value instanceof Float32Array) {

                    var val = u.value instanceof ArrayView ? u.value.dt : u.value;

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
                    if (args == pkg.textures) {

                        arg.value = val.value;
                    }
                    else {

                        arg.value = val;
                    }
                }
            }
        }

        compute(param) {
            var pkg = this.packages[param.id];
            if (!pkg) {

                pkg = this.build(param);
            }
            else {

                gl.useProgram(pkg.program); chk();
            }

            // 実引数の値をコピーする。
            this.copyParamArgsValue(param, pkg);

            // attribute変数の値をセットする。
            this.setAttribData(pkg);

            gl.useProgram(pkg.program); chk();

            // テクスチャの値のセットする。
            this.setTextureData(pkg);

            // ユニフォーム変数の値をセットする。
            this.setUniformsData(pkg);

            if (pkg.varyings.length == 0) {
                //  描画する場合

                gl.viewport(0, 0, this.canvas.width, this.canvas.height); chk();

                // カラーバッファと深度バッファをクリアする。
                gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT); chk();

                // 頂点インデックスバッファをバインドする。
                gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, pkg.VertexIndexBufferInf.buffer); chk();

                // 三角形のリストを描画する。
                gl.drawElements(gl.TRIANGLES, pkg.VertexIndexBufferInf.value.length, gl.UNSIGNED_SHORT, 0); chk();
            }
            else {
                //  描画しない場合

                // ラスタライザを無効にする。
                gl.enable(gl.RASTERIZER_DISCARD); chk();

                // Transform Feedbackをバインドする。
                gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, pkg.transformFeedback); chk();

                // すべてのvarying変数に対し
                for (var i = 0; i < pkg.varyings.length; i++) {
                    var varying = pkg.varyings[i];

                    // Transform Feedbackのバッファをバインドする。
                    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, i, varying.feedbackBuffer); chk();
                }

                // Transform Feedbackを開始する。
                gl.beginTransformFeedback(gl.POINTS); chk();    // TRIANGLES

                // 点ごとの描画をする。
                gl.drawArrays(gl.POINTS, 0, pkg.attribElementCount); chk();

                // Transform Feedbackを終了する。
                gl.endTransformFeedback(); chk();

                // ラスタライザを有効にする。
                gl.disable(gl.RASTERIZER_DISCARD); chk();

                // すべてのvarying変数に対し
                for (var i = 0; i < pkg.varyings.length; i++) {
                    varying = pkg.varyings[i];

                    // Transform Feedbackのバッファのバインドを解く。
                    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, i, null); chk();

                    // ARRAY_BUFFERにバインドする。
                    gl.bindBuffer(gl.ARRAY_BUFFER, varying.feedbackBuffer); chk();

                    var out_buf = varying.value;
                    if (out_buf instanceof ArrayView) {
                        out_buf = out_buf.dt;
                    }

                    // ARRAY_BUFFERのデータを取り出す。
                    gl.getBufferSubData(gl.ARRAY_BUFFER, 0, out_buf); chk();

                    // ARRAY_BUFFERのバインドを解く。
                    gl.bindBuffer(gl.ARRAY_BUFFER, null); chk();
                }

                // Transform Feedbackのバインドを解く。
                gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null); chk();
            }

            // プログラムの使用を終了する。
            gl.useProgram(null); chk();
        }

        drawScene() {
            var param = this.drawObj.onDraw();

            var pMatrix = mat4.create();
            mat4.perspective(45, this.canvas.width / this.canvas.height, 0.1, 100.0, pMatrix);

            var mvMatrix = mat4.create();
            mat4.identity(mvMatrix);

            mat4.translate(mvMatrix, [0.0, 0.0, this.drawParam.z]);

            mat4.rotate(mvMatrix, this.drawParam.xRot, [1, 0, 0]);
            mat4.rotate(mvMatrix, this.drawParam.yRot, [0, 1, 0]);

            var pmvMatrix = mat4.create();
            mat4.multiply(pMatrix, mvMatrix, pmvMatrix);

            var normalMatrix = mat3.create();
            mat4.toInverseMat3(mvMatrix, normalMatrix);
            mat3.transpose(normalMatrix);

            param.args["uPMVMatrix"] = pmvMatrix;
            param.args["uNMatrix"] = normalMatrix;

            this.compute(param);

            window.requestAnimationFrame(this.drawScene.bind(this));
        }

        setStandardShader() {
            this.textureSphereVertexShader = `

                const vec3 uAmbientColor = vec3(0.2, 0.2, 0.2);
                const vec3 uLightingDirection =  normalize( vec3(0.25, 0.25, 1) );
                const vec3 uDirectionalColor = vec3(0.8, 0.8, 0.8);

                // 位置
                in vec3 VertexPosition;

                // 法線
                in vec3 VertexNormal;

                // テクスチャ座標
                in vec2 TextureCoord;

                uniform mat4 uPMVMatrix;
                uniform mat3 uNMatrix;

                out vec3 vLightWeighting;

	            out vec2 uv0;
	            out vec2 uv1;

                void main(void) {
                    gl_Position = uPMVMatrix * vec4(VertexPosition, 1.0);

                    vec3 transformedNormal = uNMatrix * VertexNormal;
                    float directionalLightWeighting = max(dot(transformedNormal, uLightingDirection), 0.0);
                    vLightWeighting = uAmbientColor +uDirectionalColor * directionalLightWeighting;

		            uv0 = fract( TextureCoord.st );
		            uv1 = fract( TextureCoord.st + vec2(0.5,0.5) ) - vec2(0.5,0.5);
                }
            `;

            this.minFragmentShader =
               `out vec4 color;

                void main(){
                    color = vec4(1.0);
                }`;


            this.defaultFragmentShader =
               `in vec3 vLightWeighting;
	            in vec2 uv0;
	            in vec2 uv1;

                uniform sampler2D TextureImage;

                out vec4 color;

                void main(void) {
                    vec2 uvT;

		            uvT.x = ( fwidth( uv0.x ) < fwidth( uv1.x )-0.001 ) ? uv0.x : uv1.x ;
		            uvT.y = ( fwidth( uv0.y ) < fwidth( uv1.y )-0.001 ) ? uv0.y : uv1.y ;

                    vec4 textureColor = texture(TextureImage, uvT);

                    color = vec4(textureColor.rgb * vLightWeighting, textureColor.a);
                }
                `;
        }


        Draw3D(draw_obj) {
            this.drawObj = draw_obj;
            this.drawParam = {
                xRot : 0,
                yRot : 0,
                z    : -5.0
            }

            var lastMouseX = null;
            var lastMouseY = null;

            this.canvas.addEventListener('mousemove', function (event) {
                var newX = event.clientX;
                var newY = event.clientY;

                if (event.buttons != 0 && lastMouseX != null) {

                    this.drawParam.xRot += (newY -lastMouseY) / 300;
                    this.drawParam.yRot += (newX - lastMouseX) / 300;
                }

                lastMouseX = newX
                lastMouseY = newY;
            }.bind(this));

            this.canvas.addEventListener('touchmove', function (event) {
                // タッチによる画面スクロールを止める
                event.preventDefault(); 

                var newX = event.changedTouches[0].clientX;
                var newY = event.changedTouches[0].clientY;

                if (lastMouseX != null) {

                    this.drawParam.xRot += (newY - lastMouseY) / 300;
                    this.drawParam.yRot += (newX - lastMouseX) / 300;
                }

                lastMouseX = newX
                lastMouseY = newY;
            }.bind(this), false);

            this.canvas.addEventListener("wheel", function (e) {
                this.drawParam.z += 0.02 * e.deltaY;

                // ホイール操作によるスクロールを無効化する
                e.preventDefault();
            }.bind(this));

            this.drawScene();
        }
    }

    return new GPGPU(canvas);
}
