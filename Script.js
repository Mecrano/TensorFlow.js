function obtenerListadoArchivos(){
    var files = document.getElementById('Gatos').files;
    var imagenes = [];
    for(let i = 0; i < files.length; i++){
        imagenes.push("./Dataset/Entrenamiento/Gatos/" + files[i].name)
    }
    return imagenes;
}
function cargarImagenes(imagenes, modeloPreEntrenado ){
    console.log(imagenes);
    let promesa = Promise.resolve();
    imagenes.forEach(imagen => {
        promesa = promesa.then(data => {
            return cargarImagen(imagen).then(imagenCargada => {
                return tf.tidy(_ => {
                    const imagenProcesada = convertirImagenACuadrado(imagenCargada);
                    const prediccion = modeloPreEntrenado.predict(imagenProcesada); 

                    if(data){
                        const newData = data.concat(prediccion);
                        data.dispose();
                        return newData;
                    }

                    return tf.keep(prediccion)
                });
            });
        });
    });

    return promesa;
}
function cargarImagen(src){
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = "anonymous"; 
        img.src = src;
        img.onload = _=> resolve(tf.browser.fromPixels(img));
        img.onerror = error => reject(error);
    })
}

function convertirImagenACuadrado(img){
    //Cargar las dimensiones de nuestra imagen
    const [ancho , alto] = [img.shape[0], img.shape[1]];

    //Obtener el lado mas corto
    const ladoCorto = Math.min(ancho, alto);

    //Calcular los puntos de corte
    const initAlto = (alto - ladoCorto ) / 2;
    const initAncho = (ancho - ladoCorto ) / 2;
    const endAlto = initAlto + ladoCorto;
    const endAncho = initAncho + ladoCorto;

    //guardar la imagen cortada
    const nuevaImagen = img.slice([initAncho, initAlto, 0], [endAncho, endAlto, 3]);

    //Redimensionar nuestra imagen a 224px x 224px
    const imagen224 = tf.image.resizeBilinear(nuevaImagen, [224, 224]);

    //Expandimos nuestro tensor y traducimos los enteros en flotantes
    const loteImagen = imagen224.expandDims(0);

    return loteImagen.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
}
async function app() {
    const model = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
    return model;
}

function init(){
    app().then(modeloPreEntrenado => {
        cargarImagenes(obtenerListadoArchivos(), modeloPreEntrenado).then(data => {
            data.print();
        })
    });
}