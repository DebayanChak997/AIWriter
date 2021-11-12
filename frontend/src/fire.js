import firebase from 'firebase';
import 'firebase/auth'
import { useHistory } from "react-router";




// var firebaseConfig = {
//   apiKey: "AIzaSyBbcewE2wGGLfw19zkgLpHuN-lMO-jHlVg",
//   authDomain: "iemaiwriter-23729.firebaseapp.com",
//   databaseURL: "https://iemaiwriter-23729-default-rtdb.firebaseio.com",
//   projectId: "iemaiwriter-23729",
//   storageBucket: "iemaiwriter-23729.appspot.com",
//   messagingSenderId: "442804389721",
//   appId: "1:442804389721:web:da9b56fa72990813056e48"
//   };

  var firebaseConfig = {
    apiKey: "AIzaSyBOlwCWjEPhZzN7GtoBEAWcw4Se_aXOsXU",
    authDomain: "aiwriter-6ae7e.firebaseapp.com",
    databaseURL: "https://aiwriter-6ae7e-default-rtdb.asia-southeast1.firebasedatabase.app",
    projectId: "aiwriter-6ae7e",
    storageBucket: "aiwriter-6ae7e.appspot.com",
    messagingSenderId: "287933654027",
    appId: "1:287933654027:web:8b81b112211766864c930e",
    measurementId: "G-CHE0BPR3PQ"
  };

// var provider = new firebase.auth.GoogleAuthProvider();

// const signInWithGoogle = () => {
//   firebase.auth.signInWithPopup(provider).then((res) => {
//     console.log(res.user);
//   }).catch((err) => {
//     console.log(err);
//   })
// }



const fire = firebase.initializeApp(firebaseConfig);

const auth = fire.auth();
const db = fire.database();
const googleProvider = new firebase.auth.GoogleAuthProvider();

const signInWithGoogle = () => {
  auth.signInWithRedirect(googleProvider).then((res) => {
    console.log(res.user);
  }).catch((err) => {
    console.log(err);
  })
}
// const fire2 = firebase.initializeApp(firebaseConfig2);
export default fire;
export {
  auth,
  db,
  signInWithGoogle,};
// export {fire2};
