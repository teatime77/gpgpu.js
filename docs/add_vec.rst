
配列の加算
==========

最初に以下のような配列の加算をGPUでしてみます。

.. image:: _static/img/AddVec.png


以下はプログラムの流れです。

1. 頂点シェーダのプログラムを文字列で記述します。

  1. 入力( **in** )と出力( **out** )の変数を宣言します。
  2. **main** 関数で計算式を書きます。

2. 入力と出力の変数の値を **Float32Array** の配列で定義します。
3. GPGPUのオブジェクトを作ります。
4. 計算のパラメータを作ります。
5. パラメータを使い計算します。

以下はコードです。

.. code-block:: js

    // 頂点シェーダのプログラムを文字列で記述します。
    var vertex_shader =
       `// 入力変数A
        in vec3 A;

        // 入力変数B
        in vec3 B;

        // 出力変数C
        out vec3 C;

        // 要素ごとに呼ばれる関数。
        void main(void ) {
            C = A + B;
    }`;


    // 入力変数AをFloat32Arrayの配列で作ります。
    var A = new Float32Array([  1,  2,  3,  4,  5,  6 ]);

    // 同様に入力変数Bを作ります。
    var B = new Float32Array([10, 20, 30, 40, 50, 60]);

    // 出力変数Cは配列のサイズ(6)を指定して作ります。
    var C = new Float32Array(6);

    // 計算のパラメータ
    var param = {
        // idはプログラム内でユニークであれば何でも構いません。
        id: "AddVec",

        // 頂点シェーダの文字列を指定します。
        vertexShader: vertex_shader,

        // 頂点シェーダ内の入力と出力の変数名に値を割り当てます。
        args: {
            "A": A,
            "B": B,
            "C": C,
        }
    };

    // GPGPUのオブジェクトを作ります。
    var gpgpu = CreateGPGPU();

    // パラメータを使い計算します。
    gpgpu.compute(param);

    // 計算結果を表示します。
    document.body.insertAdjacentHTML("beforeend", "<p>C = " + C.join(' ') + "</p>");
