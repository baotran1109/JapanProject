// Import SDKs from CDN
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.8.1/firebase-app.js";
import { getAuth } from "https://www.gstatic.com/firebasejs/10.8.1/firebase-auth.js";
import { getFirestore } from "https://www.gstatic.com/firebasejs/10.8.1/firebase-firestore.js";

const firebaseConfig = {
  apiKey: "AIzaSyCiLF5j2PvkPZLzll0LX7sYRc25kYSwhoM",
  authDomain: "japan-travel-project-f2be0.firebaseapp.com",
  projectId: "japan-travel-project-f2be0",
  storageBucket: "japan-travel-project-f2be0.appspot.com",
  messagingSenderId: "587601512996",
  appId: "1:587601512996:web:931ddac27d99dee03c66b1",
  measurementId: "G-3R79HKSC5W"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export const db = getFirestore(app);
