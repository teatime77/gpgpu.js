clearAll
========

構文
^^^^^^

clearAll() 

説明
^^^^^^


WebGLのオブジェクトをすべて削除します。


ソース
^^^^^^

.. code-block:: js

        clearAll() {
            var packages = Object.assign({}, this.packages);
            for (var id in packages) {
                this.clear(id);
            }
        }


