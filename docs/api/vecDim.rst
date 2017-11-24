vecDim
======

構文
^^^^^^

vecDim(tp) 

説明
^^^^^^


ベクトルの次元を返します。


ソース
^^^^^^

.. code-block:: js

        vecDim(tp) {
            if (tp == "vec4") {
                return 4;
            }
            else if (tp == "vec3") {
                return 3;
            }
            else if (tp == "vec2") {
                return 2;
            }
            else {
                return 1;
            }
        }


