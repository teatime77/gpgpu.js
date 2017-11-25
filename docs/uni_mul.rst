
uniform変数
============

各要素の計算をするときに共通の変数の値を使いたいときは、 **uniform変数** というのを使います。

.. image:: _static/img/UniMul.png


uniform変数は以下のように **uniform** を付けて宣言します。

頂点シェーダのコード
^^^^^^^^^^^^^^^^^^^^

.. code-block:: glsl

    // 入力変数A
    in  float A;

    // uniform変数B
    uniform  float B;

    // 出力変数C
    out float C;

    // 要素ごとに呼ばれる関数。
    void main(void ) {
        C = B * A;
    }



サンプルのURL
    http://lkzf.info/gpgpu.js/samples/UniMul.html
