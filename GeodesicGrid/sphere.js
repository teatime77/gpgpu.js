﻿// JavaScript source code


function webGLStart() {
    var WebGL2;
    var gl;
    var shaderProgram;
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

    var cubeVertexPositionBuffer;
    var cubeVertexNormalBuffer;
    var cubeVertexTextureCoordBuffer;
    var cubeVertexIndexBuffer;
    var lastTime = 0;

    class Triangle {
        constructor(p, q, r, orderd) {
            if (orderd == true) {

                this.Vertexes = [p, q, r];
            }
            else {

                var a = vecSub(q, p);
                var b = vecSub(r, q);

                var c = vecCross(a, b);
                var dir = vecDot(p, c);
                if (0 < dir) {
                    this.Vertexes = [p, q, r];
                }
                else {
                    this.Vertexes = [q, p, r];
                }
            }
        }
    }

    class Vertex {
        constructor(x, y, z) {
            this.x = x;
            this.y = y;
            this.z = z;

            this.adjacentVertexes = [];
        }
    }

    class Edge {
        constructor(p1, p2) {
            this.Endpoints = [p1, p2];
        }
    }

    function sprintf() {
        var args;
        if (arguments.length == 1 && Array.isArray(arguments[0])) {

            args = arguments[0];
        }
        else {

            // 引数のリストをArrayに変換します。
            args = Array.prototype.slice.call(arguments);
        }

        switch (args.length) {
            case 0:
                console.log("");
                return;
            case 1:
                console.log("" + args[1]);
                return;
        }

        var fmt = args[0];
        var argi = 1;

        var output = "";
        var st = 0;
        var k = 0

        for (; k < fmt.length;) {
            var c1 = fmt[k];
            if (c1 = '%' && k + 1 < fmt.length) {
                var c2 = fmt[k + 1];
                if (c2 == 'd' || c2 == 'f' || c2 == 's') {

                    output += fmt.substring(st, k) + args[argi];
                    k += 2;
                    st = k;
                    argi++;
                    continue;
                }
                else if (c2 == '.' && k + 3 < fmt.length) {

                    var c3 = fmt[k + 2];
                    var c4 = fmt[k + 3];
                    if ("123456789".indexOf(c3) != -1 && c4 == 'f') {
                        var decimal_len = Number(c3);

                        var float_str = '' + args[argi];
                        var period_pos = float_str.indexOf('.');
                        if (period_pos == -1) {
                            float_str += "." + "0".repeat(decimal_len);
                        }
                        else {

                            float_str = (float_str + "0".repeat(decimal_len)).substr(0, period_pos + 1 + decimal_len);
                        }

                        output += fmt.substring(st, k) + float_str;
                        k += 4;
                        st = k;
                        argi++;
                        continue;
                    }
                }
            }
            k++;
        }
        output += fmt.substring(st, k);

        return output;
    }

    function println() {
        var output = sprintf(Array.prototype.slice.call(arguments));
        console.log(output);
    }

    function initShaders() {
        var fragmentShader = WebGL2.makeShader(gl.FRAGMENT_SHADER, fragmentShaderText);
        var vertexShader = WebGL2.makeShader(gl.VERTEX_SHADER, vertexShaderText);

        shaderProgram = gl.createProgram();
        gl.attachShader(shaderProgram, vertexShader);
        gl.attachShader(shaderProgram, fragmentShader);
        gl.linkProgram(shaderProgram);

        if (!gl.getProgramParameter(shaderProgram, gl.LINK_STATUS)) {
            alert("Could not initialise shaders");
        }

        gl.useProgram(shaderProgram);

        shaderProgram.vertexPositionAttribute = gl.getAttribLocation(shaderProgram, "aVertexPosition");
        gl.enableVertexAttribArray(shaderProgram.vertexPositionAttribute);

        shaderProgram.vertexNormalAttribute = gl.getAttribLocation(shaderProgram, "aVertexNormal");
        gl.enableVertexAttribArray(shaderProgram.vertexNormalAttribute);

        shaderProgram.textureCoordAttribute = gl.getAttribLocation(shaderProgram, "aTextureCoord");
        gl.enableVertexAttribArray(shaderProgram.textureCoordAttribute);

        shaderProgram.pMatrixUniform = gl.getUniformLocation(shaderProgram, "uPMatrix");
        shaderProgram.mvMatrixUniform = gl.getUniformLocation(shaderProgram, "uMVMatrix");
        shaderProgram.nMatrixUniform = gl.getUniformLocation(shaderProgram, "uNMatrix");
        shaderProgram.samplerUniform = gl.getUniformLocation(shaderProgram, "uSampler");
        shaderProgram.useLightingUniform = gl.getUniformLocation(shaderProgram, "uUseLighting");
        shaderProgram.ambientColorUniform = gl.getUniformLocation(shaderProgram, "uAmbientColor");
        shaderProgram.lightingDirectionUniform = gl.getUniformLocation(shaderProgram, "uLightingDirection");
        shaderProgram.directionalColorUniform = gl.getUniformLocation(shaderProgram, "uDirectionalColor");
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

    function setMatrixUniforms() {
        gl.uniformMatrix4fv(shaderProgram.pMatrixUniform, false, pMatrix);
        gl.uniformMatrix4fv(shaderProgram.mvMatrixUniform, false, mvMatrix);

        var normalMatrix = mat3.create();
        mat4.toInverseMat3(mvMatrix, normalMatrix);
        mat3.transpose(normalMatrix);
        gl.uniformMatrix3fv(shaderProgram.nMatrixUniform, false, normalMatrix);
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


    function vecLen(p) {
        return Math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
    }

    function vecDiff(p, q) {
        var dx = p.x - q.x;
        var dy = p.y - q.y;
        var dz = p.z - q.z;

        return Math.sqrt(dx * dx + dy * dy + dz * dz);
    }

    function vecSub(a, b) {
        return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
    }

    function vecDot(a, b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    function vecCross(a, b) {
        return {
            x: a.y * b.z - a.z * b.y,
            y: a.z * b.x - a.x * b.z,
            z: a.x * b.y - a.y * b.x
        };
    }

    function SetNorm(p) {
        var len = vecLen(p);

        if (len == 0) {
            p.nx = 0;
            p.ny = 0;
            p.nz = 0;
        }
        else {

            p.nx = p.x / len;
            p.ny = p.y / len;
            p.nz = p.z / len;
        }
    }

    function makeRegularIcosahedron() {
        var G = (1 + Math.sqrt(5)) / 2;

        // 頂点のリスト
        var points = [
            new Vertex( 1,  G,  0), // 0
            new Vertex( 1, -G,  0), // 1
            new Vertex(-1,  G,  0), // 2
            new Vertex(-1, -G,  0), // 3

            new Vertex( 0,  1,  G), // 4
            new Vertex( 0,  1, -G), // 5
            new Vertex( 0, -1,  G), // 6
            new Vertex( 0, -1, -G), // 7

            new Vertex( G,  0,  1), // 8
            new Vertex(-G,  0,  1), // 9
            new Vertex( G,  0, -1), // 10
            new Vertex(-G,  0, -1), // 11
        ];

        /*
0 2 4
2 0 5
0 4 8
5 0 10
0 8 10
3 1 6
1 3 7
6 1 8
1 7 10
8 1 10
4 2 9
2 5 11
9 2 11
3 6 9
7 3 11
3 9 11
4 6 8
6 4 9
7 5 10
5 7 11        
        */

        var sphere_r = vecLen(points[0]);

        points.forEach(function (x) {
            console.assert(Math.abs(sphere_r - vecLen(x)) < 0.001);
        });


        // 三角形のリスト
        var triangles = []

        for (var i1 = 0; i1 < points.length; i1++) {
            for (var i2 = i1 + 1; i2 < points.length; i2++) {
                //            println("%.2f : %d %d %.2f", sphere_r, i1, i2, vecDiff(points[i1], points[i2]));

                if (Math.abs(vecDiff(points[i1], points[i2]) - 2) < 0.01) {
                    for (var i3 = i2 + 1; i3 < points.length; i3++) {
                        if (Math.abs(vecDiff(points[i2], points[i3]) - 2) < 0.01 && Math.abs(vecDiff(points[i1], points[i3]) - 2) < 0.01) {

                            var pnts = [ points[i1], points[i2], points[i3] ]

                            var tri = new Triangle(pnts[0], pnts[1], pnts[2]);
                            for (var i = 0; i < 3; i++) {
                                pnts[i].adjacentVertexes.push(pnts[(i + 1) % 3], pnts[(i + 2) % 3])
                            }
                            
//                            println("正20面体 %d %d %d", points.indexOf(tri.Vertexes[0]), points.indexOf(tri.Vertexes[1]), points.indexOf(tri.Vertexes[2]))

                            triangles.push(tri);
                        }
                    }
                }
            }
        }
        console.assert(triangles.length == 20);

        points.forEach(function (p) {
            // 隣接する頂点の重複を取り除く。
            p.adjacentVertexes = Array.from(new Set(p.adjacentVertexes));

            console.assert(p.adjacentVertexes.length == 5);
        });

        return { points: points, triangles: triangles, sphere_r: sphere_r };
    }

    function divideTriangle(points, triangles, edges, sphere_r) {
        var divide_cnt = 4;

        for (var divide_idx = 0; divide_idx < divide_cnt; divide_idx++) {

            // 三角形を分割する。
            var new_triangles = [];

            triangles.forEach(function (x) {
                // 三角形の頂点のリスト。
                var pnts = [ x.Vertexes[0], x.Vertexes[1], x.Vertexes[2] ];

                // 中点のリスト
                var midpoints = [];

                for (var i1 = 0; i1 < 3; i1++) {

                    // 三角形の2点
                    var p1 = pnts[i1];
                    var p2 = pnts[(i1 + 1) % 3];

                    // 2点をつなぐ辺を探す。
                    var edge = edges.find(x => x.Endpoints[0] == p1 && x.Endpoints[1] == p2 || x.Endpoints[1] == p1 && x.Endpoints[0] == p2);
                    if (edge == undefined) {
                        // 2点をつなぐ辺がない場合

                        // 2点をつなぐ辺を作る。
                        edge = new Edge(p1, p2);

                        // 辺の中点を作る。
                        edge.Mid = new Vertex((p1.x + p2.x) / 2, (p1.y + p2.y) / 2, (p1.z + p2.z) / 2);

                        for (var i = 0; i < i; k++) {

                            var k = edge.Endpoints[i].adjacentVertexes.indexOf(edge.Endpoints[(i + 1) % 2]);
                            console.assert(k != -1);
                            edge.Endpoints[i].adjacentVertexes[k] = edge.Mid;
                        }

                        edges.push(edge);
                    }

                    var mid = edge.Mid;

                    midpoints.push(mid);

                    var d = vecLen(mid);
                    mid.x *= sphere_r / d;
                    mid.y *= sphere_r / d;
                    mid.z *= sphere_r / d;

                    points.push(mid);

                    console.assert(Math.abs(sphere_r - vecLen(mid)) < 0.001);
                }

                for (var i = 0; i < 3; i++) {
                    var pnt = pnts[i];
                    var mid = midpoints[i];

                    if (mid.adjacentVertexes.length == 0) {

                        mid.adjacentVertexes.push(pnts[(i + 1) % 3], midpoints[(i + 1) % 3], midpoints[(i + 2) % 3], pnts[i]);
                    }
                    else {

                        mid.adjacentVertexes.push(pnts[(i + 1) % 3], midpoints[(i + 2) % 3]);
                    }
                }

                new_triangles.push(new Triangle(midpoints[0], midpoints[1], midpoints[2], true));
                new_triangles.push(new Triangle(pnts[0], midpoints[0], midpoints[2], true));
                new_triangles.push(new Triangle(pnts[1], midpoints[1], midpoints[0], true));
                new_triangles.push(new Triangle(pnts[2], midpoints[2], midpoints[1], true));
            });

            points.forEach(function (p) {
                console.assert(p.adjacentVertexes.length == 5 || p.adjacentVertexes.length == 6);
            });

            triangles = new_triangles;
        }

        /*

        var new_triangles = [];
        triangles.forEach(function (x) {
            if (x.Vertexes.every(p => p.adjacentVertexes.length == 6)) {
                new_triangles.push(x);
            }
        });
        triangles = new_triangles;
        */

        for (var i = 0; i < points.length; i++) {
            var p = points[i];
            console.assert(i < 12 && p.adjacentVertexes.length == 5 || p.adjacentVertexes.length == 6);

            var x = p.z / sphere_r;
            var y = p.x / sphere_r;
            var z = p.y / sphere_r;

            var th = Math.asin(z);  // [-PI/2 , PI/2]

            p.texY = Math.min(1, Math.max(0, th / Math.PI + 0.5));
//            p.texX = 0.5;
//            continue;

            var r = Math.cos(th);

            if (r == 0) {

                p.texX = 0;
                continue;
            }

            x /= r;
            y /= r;

            var ph = Math.atan2(y, x);  // [-PI , PI]

            var u = ph / Math.PI;

            p.texX = Math.min(1, Math.max(0, ph / (2 * Math.PI) + 0.5));
        }

        println("半径:%.3f 三角形 %d", sphere_r, triangles.length);

        return triangles;
    }

    function initBuffers() {
        var ret           = makeRegularIcosahedron();
        var points        = ret.points;
        var triangles = ret.triangles;
        var sphere_r = ret.sphere_r;

        var edges = [];

        triangles = divideTriangle(points, triangles, edges, sphere_r);

        // 頂点インデックス
        var vertexIndices = [];

        triangles.forEach(x =>
            vertexIndices.push(points.indexOf(x.Vertexes[0]), points.indexOf(x.Vertexes[1]), points.indexOf(x.Vertexes[2]))
        );

        // 法線をセット
        points.forEach(p => SetNorm(p));

        // 位置の配列
        var vertices = [];
        points.forEach(p =>
            vertices.push(p.x, p.y, p.z)
        );

        // 法線の配列
        var vertexNormals = [];
        points.forEach(p =>
            vertexNormals.push(p.nx, p.ny, p.nz)
        );

        // テクスチャ座標
        var textureCoords = [];
        points.forEach(p =>
            textureCoords.push(p.texX, p.texY)
        );

        cubeVertexPositionBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, cubeVertexPositionBuffer);

        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
        cubeVertexPositionBuffer.itemSize = 3;
        cubeVertexPositionBuffer.numItems = points.length;

        cubeVertexNormalBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, cubeVertexNormalBuffer);

        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertexNormals), gl.STATIC_DRAW);
        cubeVertexNormalBuffer.itemSize = 3;
        cubeVertexNormalBuffer.numItems = points.length;

        cubeVertexTextureCoordBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, cubeVertexTextureCoordBuffer);

        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(textureCoords), gl.STATIC_DRAW);
        cubeVertexTextureCoordBuffer.itemSize = 2;
        cubeVertexTextureCoordBuffer.numItems = points.length;

        cubeVertexIndexBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, cubeVertexIndexBuffer);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(vertexIndices), gl.STATIC_DRAW);
        cubeVertexIndexBuffer.itemSize = 1;
        cubeVertexIndexBuffer.numItems = vertexIndices.length;
    }

    function drawScene() {
        gl.viewport(0, 0, gl.viewportWidth, gl.viewportHeight);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

        mat4.perspective(45, gl.viewportWidth / gl.viewportHeight, 0.1, 100.0, pMatrix);

        mat4.identity(mvMatrix);

        mat4.translate(mvMatrix, [0.0, 0.0, z]);

        mat4.rotate(mvMatrix, degToRad(xRot), [1, 0, 0]);
        mat4.rotate(mvMatrix, degToRad(yRot), [0, 1, 0]);

        gl.bindBuffer(gl.ARRAY_BUFFER, cubeVertexPositionBuffer);
        gl.vertexAttribPointer(shaderProgram.vertexPositionAttribute, cubeVertexPositionBuffer.itemSize, gl.FLOAT, false, 0, 0);

        gl.bindBuffer(gl.ARRAY_BUFFER, cubeVertexNormalBuffer);
        gl.vertexAttribPointer(shaderProgram.vertexNormalAttribute, cubeVertexNormalBuffer.itemSize, gl.FLOAT, false, 0, 0);

        gl.bindBuffer(gl.ARRAY_BUFFER, cubeVertexTextureCoordBuffer);
        gl.vertexAttribPointer(shaderProgram.textureCoordAttribute, cubeVertexTextureCoordBuffer.itemSize, gl.FLOAT, false, 0, 0);

        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, crateTexture);
        gl.uniform1i(shaderProgram.samplerUniform, 0);
        var lighting = document.getElementById("lighting").checked;
        gl.uniform1i(shaderProgram.useLightingUniform, lighting);
        if (lighting) {
            gl.uniform3f(
                shaderProgram.ambientColorUniform,
                parseFloat(document.getElementById("ambientR").value),
                parseFloat(document.getElementById("ambientG").value),
                parseFloat(document.getElementById("ambientB").value)
            );

            var lightingDirection = [
                parseFloat(document.getElementById("lightDirectionX").value),
                parseFloat(document.getElementById("lightDirectionY").value),
                parseFloat(document.getElementById("lightDirectionZ").value)
            ];
            var adjustedLD = vec3.create();
            vec3.normalize(lightingDirection, adjustedLD);
            vec3.scale(adjustedLD, -1);
            gl.uniform3fv(shaderProgram.lightingDirectionUniform, adjustedLD);

            gl.uniform3f(
                shaderProgram.directionalColorUniform,
                parseFloat(document.getElementById("directionalR").value),
                parseFloat(document.getElementById("directionalG").value),
                parseFloat(document.getElementById("directionalB").value)
            );
        }

        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, cubeVertexIndexBuffer);
        setMatrixUniforms();
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

        drawScene();
        animate();
    }

    var time = new Date();
    var hh = time.getHours();
    var mm = time.getMinutes();
    var ss = time.getSeconds();
    console.log("" + hh + "時" + mm + "分" + ss + "秒をお知らせします。")

    var canvas = document.getElementById("lesson07-canvas");
    WebGL2 = CreateWebGLLib(canvas);
    gl = WebGL2.getGL();
    gl.viewportWidth = canvas.width;
    gl.viewportHeight = canvas.height;

    initShaders();
    initBuffers();
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
