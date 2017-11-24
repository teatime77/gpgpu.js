setAttribData
=============

構文
^^^^^^

setAttribData(pkg) 

説明
^^^^^^


attribute変数のデータをセットします。


ソース
^^^^^^

.. code-block:: js

        setAttribData(pkg) {
            // すべてのattribute変数に対し
            for (let attrib of pkg.attributes) {
                var dim = this.vecDim(attrib.type);

                gl.bindBuffer(gl.ARRAY_BUFFER, attrib.AttribBuffer); chk();

                // 指定した位置のattribute変数の要素数(dim)と型(float)をセットする。
                gl.vertexAttribPointer(attrib.AttribLoc, dim, gl.FLOAT, false, 0, 0); chk();

                // attribute変数のデータをセットする。
                gl.bufferData(gl.ARRAY_BUFFER, attrib.value, gl.STATIC_DRAW);
            }
        }


