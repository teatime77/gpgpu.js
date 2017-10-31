// JavaScript source code

var MyWebGL;

function webGLStart() {
    var xRot = 0;
    var yRot = 0;
    var z = -5.0;

    var lastMouseX = null;
    var lastMouseY = null;

    var param;

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

    function drawScene() {

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

        MyWebGL.compute(param);
    }

    function tick() {
        requestAnimFrame(tick);
        drawScene();
    }

    var canvas = document.getElementById("lesson07-canvas");
    MyWebGL = CreateWebGLLib(canvas);

    var img = new Image();
    img.onload = function () {

        param = initBuffers(img);

        tick();
    }

    img.src = "world.topo.bathy.200408.2048x2048.png";// "earth.png";// "crate.gif";

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
}


function initBuffers(img) {
    var ret = makeEarthBuffers();

    var param = {
        id: "Earth",
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
            "uSampler": img,
        }
        ,
        VertexIndexBuffer: ret.idx_array
    };

    return param;
}