makeAttrib
==========

構文
^^^^^^

makeAttrib(pkg) 

説明
^^^^^^


attribute変数を作ります。


ソース
^^^^^^

.. code-block:: js

        makeAttrib(pkg) {
            // すべてのattribute変数に対し
            for (let attrib of pkg.attributes) {
                // attribute変数の次元
                var attrib_dim = this.vecDim(attrib.type);

                // 要素の個数
                var elemen_count = attrib.value.length / attrib_dim;

                if (pkg.elementCount == undefined) {
                    pkg.attribElementCount = elemen_count;
                }
                else {

                    assert(pkg.elementCount == elemen_count);
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


