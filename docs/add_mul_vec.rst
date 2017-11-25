
配列の加算と乗算
================

今度は加算と乗算の2つの値を出力してみます。

.. image:: _static/img/AddMulVec.png


2個以上の変数に出力するには前回のコードに出力変数の記述を追加するだけです。

頂点シェーダのコード
^^^^^^^^^^^^^^^^^^^^

.. code-block:: glsl

    // 入力変数A
    in  float A;

    // 入力変数B
    in  float B;

    // 出力変数C
    out float C;

    // 出力変数D
    out float D;

    // 要素ごとに呼ばれる関数。
    void main(void ) {
        C = A + B;
        D = A * B;
    }


1つ注意すべき点は、 **入力変数と出力変数の要素の数はすべて同じ** というきまりがあることです。
この例ではA,B,C,Dの各配列の要素数はすべて6になっています。


サンプルのURL
    http://lkzf.info/gpgpu.js/samples/AddMulVec.html
