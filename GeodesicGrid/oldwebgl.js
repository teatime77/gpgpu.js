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

        MyWebGL.setAttribData(pkg);

        // テクスチャの値のセット
        MyWebGL.setTextureData(pkg);

        MyWebGL.copyParamArgsValue(param, pkg);

        MyWebGL.setUniformsData(pkg);
        MyWebGL.draw(pkg);
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
        param = initBuffers(pkg);

        // テクスチャの初期処理
        MyWebGL.makeTexture(pkg);
        imageLoaded = true;
    }

    texImg.src = "world.topo.bathy.200408.2048x2048.png";// "earth.png";// "crate.gif";

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


function initBuffers(pkg) {
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
        VertexIndexBuffer: ret.idx_array
    };

    MyWebGL.parseShader(pkg, param);
    MyWebGL.makeAttrib(pkg);
    MyWebGL.makeVertexIndexBuffer(pkg, param);

    // ユニフォーム変数の初期処理
    MyWebGL.initUniform(pkg);

    return param;
}