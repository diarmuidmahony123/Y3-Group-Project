import { initializeApp } from 'firebase/app'
import{getDatabase,ref,set} from 'firebase/database'

import { getAuth, createUserWithEmailAndPassword } from 'firebase/auth';




const firebaseConfig = {
  apiKey: "AIzaSyBcR46Vvrz7x-32jCcpHUQBl3o03W7lQio",
  authDomain: "year3groupproject-27054.firebaseapp.com",
  databaseURL: "https://year3groupproject-27054-default-rtdb.europe-west1.firebasedatabase.app",
  projectId: "year3groupproject-27054",
  storageBucket: "year3groupproject-27054.appspot.com",
  messagingSenderId: "628491120267",
  appId: "1:628491120267:web:c85f021b55f1b055eeae8e",
  measurementId: "G-36YZWBW5BQ"
};





const app = initializeApp(firebaseConfig);

const auth = getAuth(app);

createUserWithEmailAndPassword(auth, email, password).then((userCredential)=> {
  const user = userCredential.user;


})
.catch((error) => {
  const errorCode = error.code;
  const errorMessage = error.message;
})

