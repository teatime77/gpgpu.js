// JavaScript source code

var MyWebGL;

var cubeVertexPositionBuffer;
var cubeVertexNormalBuffer;
var cubeVertexTextureCoordBuffer;
var cubeVertexIndexBuffer;
var texImg;

function webGLStart() {
    var gl;
    var imageLoaded = false;

    var xRot = 0;
    var yRot = 0;
    var z = -5.0;

    var lastMouseX = null;
    var lastMouseY = null;

    var param;
    var pkg = {};
    var tex_inf = {};

    function initShaders(pkg) {
        var vertex_shader = MyWebGL.makeShader(gl.VERTEX_SHADER, vertexShaderText);
        var fragmentShader = MyWebGL.makeShader(gl.FRAGMENT_SHADER, fragmentShaderText);

        pkg.program = MyWebGL.makeProgram(vertex_shader, fragmentShader);

        gl.useProgram(pkg.program);

    }

    function handleLoadedTexture() {
        tex_inf.locTexture = gl.getUniformLocation(pkg.program, "uSampler");
        tex_inf.Texture = gl.createTexture();

        gl.bindTexture(gl.TEXTURE_2D, tex_inf.Texture);

        gl.bindTexture(gl.TEXTURE_2D, null);
    }

    function degToRad(degrees) {
        return degrees * Math.PI / 180;
    }


    function handleMouseMove(event) {
        var newX = event.clientX;
        var newY = event.clientY;

        if (event.buttons != 0 && lastMouseX != null) {

            xRot += (newY - lastMouseY) / 5;
            yRot += (newX - lastMouseX) / 5;
        }

        lastMouseX = newX
        lastMouseY = newY;
    }

    function drawScene(pkg) {

        var pMatrix = mat4.create();
        mat4.perspective(45, MyWebGL.canvas.width / MyWebGL.canvas.height, 0.1, 100.0, pMatrix);

        var mvMatrix = mat4.create();
        mat4.identity(mvMatrix);

        mat4.translate(mvMatrix, [0.0, 0.0, z]);

        mat4.rotate(mvMatrix, degToRad(xRot), [1, 0, 0]);
        mat4.rotate(mvMatrix, degToRad(yRot), [0, 1, 0]);

        var lighting = document.getElementById("lighting").checked;

        var ambient = new Float32Array([
            parseFloat(document.getElementById("ambientR").value),
            parseFloat(document.getElementById("ambientG").value), 
            parseFloat(document.getElementById("ambientB").value)
        ]);
        var directionalColor = new Float32Array([
            parseFloat(document.getElementById("directionalR").value),
            parseFloat(document.getElementById("directionalG").value),
            parseFloat(document.getElementById("directionalB").value)
        ]);

        var lightingDirection = new Float32Array([
            parseFloat(document.getElementById("lightDirectionX").value),
            parseFloat(document.getElementById("lightDirectionY").value),
            parseFloat(document.getElementById("lightDirectionZ").value)
        ]);
        var adjustedLD = vec3.create();
        vec3.normalize(lightingDirection, adjustedLD);
        vec3.scale(adjustedLD, -1);

        var normalMatrix = mat3.create();
        mat4.toInverseMat3(mvMatrix, normalMatrix);
        mat3.transpose(normalMatrix);

        param.args["uUseLighting"]       = lighting;
        param.args["uAmbientColor"]      = ambient;
        param.args["uDirectionalColor"]  = directionalColor;
        param.args["uLightingDirection"] = adjustedLD;
        param.args["uPMatrix"]           = pMatrix;
        param.args["uMVMatrix"]          = mvMatrix;
        param.args["uNMatrix"]           = normalMatrix;

        gl.viewport(0, 0, MyWebGL.canvas.width, MyWebGL.canvas.height);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

        MyWebGL.setAttribData(pkg);

        /*
        gl.uniform1i(tex_inf.locTexture, 0);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, tex_inf.Texture);
        */

        // テクスチャの値のセット
        MyWebGL.setTextureData(pkg);


        MyWebGL.copyParamArgsValue(param, pkg);

        MyWebGL.setUniformsData(pkg);

        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, cubeVertexIndexBuffer);
        gl.drawElements(gl.TRIANGLES, cubeVertexIndexBuffer.numItems, gl.UNSIGNED_SHORT, 0);
    }

    function tick() {
        requestAnimFrame(tick);
        if (!imageLoaded) {
            return;
        }

        drawScene(pkg);
    }

    var time = new Date();
    var hh = time.getHours();
    var mm = time.getMinutes();
    var ss = time.getSeconds();
    console.log("" + hh + "時" + mm + "分" + ss + "秒をお知らせします。")

    var canvas = document.getElementById("lesson07-canvas");
    MyWebGL = CreateWebGLLib(canvas);
    gl = MyWebGL.getGL();


    texImg = new Image();
    texImg.onload = function () {

        initShaders(pkg);
        param = initBuffers(pkg, gl);

        // テクスチャの初期処理
        MyWebGL.makeTexture(pkg);
        //            handleLoadedTexture();
        imageLoaded = true;
    }

    texImg.src = "world.topo.bathy.200408.2048x2048.png";// "earth.png";// "crate.gif";



    gl.clearColor(0.0, 0.0, 0.0, 1.0);
    gl.enable(gl.DEPTH_TEST);


    document.onmousemove = handleMouseMove;

    if (window.WheelEvent) {

        // ------------------------------------------------------------
        // ホイールを操作すると実行されるイベント
        // ------------------------------------------------------------
        document.addEventListener("wheel", function (e) {
            z += 0.002 * e.wheelDelta;

            // ホイール操作によるスクロールを無効化する
            e.preventDefault();
        });
    }

    tick();
}


function initBuffers(pkg, gl) {
    var ret = makeEarthBuffers();

    var param = {
        vertexShader: vertexShaderText,
        fragmentShader: fragmentShaderText
        ,
        args: {
            "aVertexPosition": ret.vertex_array,
            "aVertexNormal": ret.normal_array,
            "aTextureCoord": ret.texture_array,

            "uUseLighting": 1,
            "uAmbientColor": 1,
            "uDirectionalColor": 1,
            "uLightingDirection": 1,
            "uPMatrix": 1,
            "uMVMatrix": 1,
            "uNMatrix": 1,
            "uSampler": texImg,
            /*
            "vTextureCoord": 1,
            "vLightWeighting": 1,
            "uv0": 1,
            "uv1": 1,
            */
        }
        ,
        fixed: [

        ]
    };

    MyWebGL.parseShader(pkg, param);
    MyWebGL.makeAttrib(pkg);

    cubeVertexIndexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, cubeVertexIndexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, ret.idx_array, gl.STATIC_DRAW);
    cubeVertexIndexBuffer.itemSize = 1;
    cubeVertexIndexBuffer.numItems = ret.idx_array.length;

    // ユニフォーム変数の初期処理
    MyWebGL.initUniform(pkg);

    return param;
}