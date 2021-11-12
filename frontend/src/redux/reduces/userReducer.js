const initialValue = {
    serverError: false,
    isAuthenticate: false,
    loading: false,
 }
export const userLoginReducer = (state={initialValue}, action) =>{
    switch (action.type) {
        case "USER_LOGIN_SUCCESS":
            return{
                serverError: false,
                isAuthenticate: true,
                loading: false,
                adminInfo: action.payload,
            }
        case "ADMIN_LOGOUT":
            return{
                    isAuthenticate: false,
                }
            
    
        default:
            return state;
    }
  }