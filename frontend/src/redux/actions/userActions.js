export const userLogin = (userInfo) => async (dispatch) =>{
    try {
        dispatch({
            type:"USER_LOGIN_SUCCESS",
            payload:userInfo
            
        })
        window.sessionStorage.setItem("user", JSON.stringify(userInfo));
    } catch (error) {
        dispatch({
            type:"USER_LOGIN_FAIELD",
            payload: error.message
        })
    }
}

// logout action
export const userLogout = () => (dispatch) => {
    window.sessionStorage.removeItem("user");
    dispatch({
      type: "USER_LOGOUT",
    });
  };