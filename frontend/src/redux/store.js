import {createStore,applyMiddleware, combineReducers} from "redux";
import { composeWithDevTools } from "redux-devtools-extension";
import thunk from "redux-thunk";

//import all reducers
import {userLoginReducer} from "./reduces/userReducer"




const rootReducer = combineReducers({
    userLogin:userLoginReducer
})



//getting User data from local storage
const userDetails = JSON.parse(window.sessionStorage.getItem("user"));

const initialState = {
    userLogin:{
        userDetails: userDetails
    },
}

const middleware = [thunk]

//Creating store
const store = createStore(
    rootReducer,
    initialState,
    composeWithDevTools(applyMiddleware(...middleware))
)

export default store;