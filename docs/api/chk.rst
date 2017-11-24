chk
===

構文
^^^^^^

chk() 

説明
^^^^^^


WebGLのエラーのチェックをします。


ソース
^^^^^^

.. code-block:: js

    function chk() {
        assert(gl.getError() == gl.NO_ERROR);
    }


