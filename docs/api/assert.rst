assert
======

構文
^^^^^^

assert(condition, message) 

説明
^^^^^^


エラーのチェックをします。


ソース
^^^^^^

.. code-block:: js

    function assert(condition, message) {
        if (!condition) {
            throw new Error(message || "Assertion failed");
        }
    }


