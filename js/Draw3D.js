// JavaScript source code

function webGLStart() {
    var xRot = 0;
    var yRot = 0;
    var z = -5.0;

    var lastMouseX = null;
    var lastMouseY = null;

    var param;

    var MyWebGL;

    var vertex_shader = `#version 300 es

        precision highp float;
        precision highp int;

        // 位置
        in vec3 aVertexPosition;

        // 法線
        in vec3 aVertexNormal;

        // テクスチャ座標
        in vec2 aTextureCoord;

        uniform mat4 uMVMatrix;
        uniform mat4 uPMatrix;
        uniform mat3 uNMatrix;

        uniform vec3 uAmbientColor;

        uniform vec3 uLightingDirection;
        uniform vec3 uDirectionalColor;

        uniform bool uUseLighting;

        out vec3 vLightWeighting;

	    out vec2 uv0;
	    out vec2 uv1;

        void main(void) {
            gl_Position = uPMatrix * uMVMatrix * vec4(aVertexPosition, 1.0);

            if (!uUseLighting) {
                vLightWeighting = vec3(1.0, 1.0, 1.0);
            } else {
                vec3 transformedNormal = uNMatrix * aVertexNormal;
                float directionalLightWeighting = max(dot(transformedNormal, uLightingDirection), 0.0);
                vLightWeighting = uAmbientColor + uDirectionalColor * directionalLightWeighting;
            }

		    uv0 = fract( aTextureCoord.st );
		    uv1 = fract( aTextureCoord.st + vec2(0.5,0.5) ) - vec2(0.5,0.5);
        }
    `;

    var fragmentShaderText = `#version 300 es

        precision highp float;
        precision highp int;

        in vec3 vLightWeighting;
	    in vec2 uv0;
	    in vec2 uv1;

        uniform sampler2D uSampler;

        out vec4 color;

        void main(void) {
            vec2 uvT;

		    uvT.x = ( fwidth( uv0.x ) < fwidth( uv1.x )-0.001 ) ? uv0.x : uv1.x ;
		    uvT.y = ( fwidth( uv0.y ) < fwidth( uv1.y )-0.001 ) ? uv0.y : uv1.y ;

            vec4 textureColor = texture(uSampler, uvT);

            color = vec4(textureColor.rgb * vLightWeighting, textureColor.a);
        }
    `;

    function initBuffers(img) {
        var ret = makeEarthBuffers();

        var param = {
            id: "Earth",
            vertexShader: vertex_shader,
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

    function drawScene() {

        var pMatrix = mat4.create();
        mat4.perspective(45, MyWebGL.canvas.width / MyWebGL.canvas.height, 0.1, 100.0, pMatrix);

        var mvMatrix = mat4.create();
        mat4.identity(mvMatrix);

        mat4.translate(mvMatrix, [0.0, 0.0, z]);

        mat4.rotate(mvMatrix, xRot, [1, 0, 0]);
        mat4.rotate(mvMatrix, yRot, [0, 1, 0]);

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

        requestAnimFrame(drawScene);
    }

    var canvas = document.getElementById("lesson07-canvas");
    MyWebGL = CreateWebGLLib(canvas);

    var img = new Image();
    img.onload = function () {

        param = initBuffers(img);

        drawScene();
    }

    img.src = "world.topo.bathy.200408.2048x2048.png";// "earth.png";// "crate.gif";

    canvas.addEventListener('mousemove', function (event) {
        var newX = event.clientX;
        var newY = event.clientY;

        if (event.buttons != 0 && lastMouseX != null) {

            xRot += (newY - lastMouseY) / 300;
            yRot += (newX - lastMouseX) / 300;
        }

        lastMouseX = newX
        lastMouseY = newY;
    });

    canvas.addEventListener("wheel", function (e) {
        z += 0.002 * e.wheelDelta;

        // ホイール操作によるスクロールを無効化する
        e.preventDefault();
    });
}
