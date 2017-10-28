// JavaScript source code

var MyWebGL;

var cubeVertexPositionBuffer;
var cubeVertexNormalBuffer;
var cubeVertexTextureCoordBuffer;
var cubeVertexIndexBuffer;

function webGLStart() {
    var gl;
    var crateTexture;
    var imageLoaded = false;
    var mvMatrix = mat4.create();
    var pMatrix = mat4.create();
    var xRot = 0;

    var yRot = 0;

    var z = -5.0;

    var lastMouseX = null;
    var lastMouseY = null;

    var moonRotationMatrix = mat4.create();
    mat4.identity(moonRotationMatrix);
    var lastTime = 0;
    var param;
    var pkg = {};

    function initShaders(pkg) {
        var vertex_shader = MyWebGL.makeShader(gl.VERTEX_SHADER, vertexShaderText);
        var fragmentShader = MyWebGL.makeShader(gl.FRAGMENT_SHADER, fragmentShaderText);

        pkg.program = MyWebGL.makeProgram(vertex_shader, fragmentShader);

        gl.useProgram(pkg.program);

        pkg.program.samplerUniform = gl.getUniformLocation(pkg.program, "uSampler");
    }

    function handleLoadedTexture(texture) {
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);

        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, texture.image);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_NEAREST);

//        gl.texParameteri(gl.TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_MIRRORED_REPEAT); //GL_REPEAT
//        gl.texParameteri(gl.TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_MIRRORED_REPEAT); //GL_REPEAT

        gl.generateMipmap(gl.TEXTURE_2D);

        gl.bindTexture(gl.TEXTURE_2D, null);
    }

    function initTexture() {
        crateTexture = gl.createTexture();
        crateTexture.image = new Image();
        crateTexture.image.onload = function () {
            handleLoadedTexture(crateTexture);
            imageLoaded = true;
        }

        crateTexture.image.src = "world.topo.bathy.200408.2048x2048.png";// "earth.png";// "crate.gif";
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
        gl.viewport(0, 0, gl.viewportWidth, gl.viewportHeight);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

        mat4.perspective(45, gl.viewportWidth / gl.viewportHeight, 0.1, 100.0, pMatrix);

        mat4.identity(mvMatrix);

        mat4.translate(mvMatrix, [0.0, 0.0, z]);

        mat4.rotate(mvMatrix, degToRad(xRot), [1, 0, 0]);
        mat4.rotate(mvMatrix, degToRad(yRot), [0, 1, 0]);

        MyWebGL.setAttribData(pkg);

        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, crateTexture);
        gl.uniform1i(pkg.program.samplerUniform, 0);

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

        MyWebGL.copyParamArgsValue(param, pkg);

        MyWebGL.setUniformsData(pkg);

        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, cubeVertexIndexBuffer);
        gl.drawElements(gl.TRIANGLES, cubeVertexIndexBuffer.numItems, gl.UNSIGNED_SHORT, 0);
    }

    function animate() {
        var timeNow = new Date().getTime();
        if (lastTime != 0) {
            var elapsed = timeNow - lastTime;
        }
        lastTime = timeNow;
    }


    function tick() {
        requestAnimFrame(tick);
        if (!imageLoaded) {
            return;
        }

        drawScene(pkg);
        animate();
    }

    var time = new Date();
    var hh = time.getHours();
    var mm = time.getMinutes();
    var ss = time.getSeconds();
    console.log("" + hh + "時" + mm + "分" + ss + "秒をお知らせします。")

    var canvas = document.getElementById("lesson07-canvas");
    MyWebGL = CreateWebGLLib(canvas);
    gl = MyWebGL.getGL();
    gl.viewportWidth = canvas.width;
    gl.viewportHeight = canvas.height;

    initShaders(pkg);
    param = initBuffers(pkg, gl);
    initTexture();

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
